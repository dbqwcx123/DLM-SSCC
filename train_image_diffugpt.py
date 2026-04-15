import warnings
from requests.exceptions import RequestsDependencyWarning
warnings.filterwarnings("ignore", category=RequestsDependencyWarning)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 关闭 tokenizers 的并行以避免警告
import sys
import argparse
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, TrainingArguments, TrainerCallback
from torch.utils.data import IterableDataset, get_worker_info
import torch.utils.checkpoint
from functools import partial
import torch.distributed as dist
import math
import imageio
from natsort import natsorted
import wandb
from datetime import datetime
import random

import constants
from diffu_model import *
from diffu_trainer import ImageDiscreteDiffusionTrainer
from llamafactory.hparams import FinetuningArguments


class Div2kPatchDataset(IterableDataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        num_chunks=-1,
        is_channel_wised=False,
        shuffle=True,
        split='train',
        val_ratio=0.1,
        split_seed=42
    ):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.num_chunks = num_chunks
        self.is_channel_wised = is_channel_wised
        self.shuffle = shuffle
        self.split = split

        # 给 callback / trainer 用
        self.base_seed = 42
        self.epoch = 0

        if not os.path.exists(data_path):
            raise ValueError(f"Data path {data_path} does not exist.")

        all_files = [
            os.path.join(data_path, item)
            for item in os.listdir(data_path)
            if item.lower().endswith('.png')
        ]
        all_files = natsorted(all_files)

        if len(all_files) == 0:
            raise ValueError(f"No PNG files found in {data_path}")

        # 固定划分 train / valid
        split_rng = random.Random(split_seed)
        split_perm = list(all_files)
        split_rng.shuffle(split_perm)

        num_valid = max(1, int(len(split_perm) * val_ratio))
        num_train = len(split_perm) - num_valid

        if split == 'train':
            self.all_files = split_perm[:num_train]
        elif split == 'valid':
            self.all_files = split_perm[num_train:]
        else:
            self.all_files = split_perm

        # 统计当前 split 的真实 patch 数
        h, w = constants.CHUNK_SHAPE_2D
        total_patches = 0
        for file_path in self.all_files:
            image = imageio.v2.imread(file_path)
            height, width = image.shape[0], image.shape[1]

            patches_per_image = (height // h) * (width // w)
            if self.is_channel_wised:
                if image.ndim == 2:
                    num_channels = 1
                else:
                    num_channels = image.shape[-1]
                patches_per_image *= num_channels

            total_patches += patches_per_image

        self.total_patches = total_patches
        self.effective_total_patches = (
            min(self.num_chunks, self.total_patches)
            if self.num_chunks > 0 else self.total_patches
        )

        print(
            f"Dataset initialized: split={split}, "
            f"images={len(self.all_files)}, "
            f"total_patches={self.total_patches}, "
            f"effective_patches={self.effective_total_patches}"
        )

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def process_patch_to_tokens(self, patch):
        flat_pixels = patch.flatten()
        num_str_tokens = [str(val) for val in flat_pixels]
        input_ids = [self.tokenizer.bos_token_id] + self.tokenizer.convert_tokens_to_ids(num_str_tokens)
        return torch.tensor(input_ids, dtype=torch.long)

    def _iter_patches_from_file(self, file_path):
        """
        边切 patch 边 yield，不再先构造 patches = []
        """
        h, w = constants.CHUNK_SHAPE_2D
        image = imageio.v2.imread(file_path)
        height, width = image.shape[0], image.shape[1]

        if self.is_channel_wised:
            num_channels = 1 if image.ndim == 2 else image.shape[-1]
            patch_descs = []
            for ch in range(num_channels):
                for row in range(height // h):
                    for col in range(width // w):
                        patch_descs.append((row, col, ch))

            # 每张图内部 patch 顺序也可复现打乱，减少文件级块状偏置
            if self.shuffle:
                local_rng = random.Random((hash(file_path) ^ self.base_seed ^ self.epoch) & 0xffffffff)
                local_rng.shuffle(patch_descs)

            for row, col, ch in patch_descs:
                r0, r1 = row * h, (row + 1) * h
                c0, c1 = col * w, (col + 1) * w
                if image.ndim == 2:
                    patch = image[r0:r1, c0:c1][..., None]
                else:
                    patch = image[r0:r1, c0:c1, ch:ch+1]
                yield patch

        else:
            patch_descs = []
            for row in range(height // h):
                for col in range(width // w):
                    patch_descs.append((row, col))

            if self.shuffle:
                local_rng = random.Random((hash(file_path) ^ self.base_seed ^ self.epoch) & 0xffffffff)
                local_rng.shuffle(patch_descs)

            for row, col in patch_descs:
                r0, r1 = row * h, (row + 1) * h
                c0, c1 = col * w, (col + 1) * w
                patch = image[r0:r1, c0:c1]
                yield patch

    def _get_image_iterator(self, files, max_items=-1):
        """
        max_items 是当前 worker 实际最多需要产出的 patch 数。
        """
        produced = 0

        for file_path in files:
            for patch in self._iter_patches_from_file(file_path):
                if max_items > 0 and produced >= max_items:
                    return

                input_ids = self.process_patch_to_tokens(patch)
                src_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                if input_ids[0] == self.tokenizer.bos_token_id:
                    src_mask[0] = True

                yield {
                    "input_ids": input_ids,
                    "src_mask": src_mask
                }
                produced += 1

    def __iter__(self):
        # 按 epoch 生成当前轮次的文件顺序
        if self.shuffle:
            rng = random.Random(self.base_seed + self.epoch)
            ordered_files = list(self.all_files)
            rng.shuffle(ordered_files)
        else:
            ordered_files = self.all_files

        # --- DDP rank 切分 ---
        if dist.is_initialized():
            num_gpus = dist.get_world_size()
            gpu_id = dist.get_rank()
        else:
            num_gpus = 1
            gpu_id = 0

        per_gpu_files_count = int(math.ceil(len(ordered_files) / float(num_gpus)))
        start_gpu = gpu_id * per_gpu_files_count
        end_gpu = min(start_gpu + per_gpu_files_count, len(ordered_files))
        files_on_this_gpu = ordered_files[start_gpu:end_gpu]

        # --- 单卡内 worker 切分 ---
        worker_info = get_worker_info()
        if worker_info is None:
            files_to_process = files_on_this_gpu
            local_num_workers = 1
            local_worker_id = 0
        else:
            per_worker = int(math.ceil(len(files_on_this_gpu) / float(worker_info.num_workers)))
            local_worker_id = worker_info.id
            local_num_workers = worker_info.num_workers
            start = local_worker_id * per_worker
            end = min(start + per_worker, len(files_on_this_gpu))
            files_to_process = files_on_this_gpu[start:end]

        # --- 对 num_chunks 做近似均分截断 ---
        # 这里仍然是“尽量少改动”的版本，不做精确全局 patch 区间映射，
        # 但至少避免每个 worker 都各自产 num_chunks 个样本
        if self.effective_total_patches > 0:
            if dist.is_initialized():
                world_size = dist.get_world_size()
            else:
                world_size = 1

            total_streams = world_size * local_num_workers
            per_stream_items = int(math.ceil(self.effective_total_patches / float(total_streams)))
        else:
            per_stream_items = -1

        return self._get_image_iterator(files_to_process, max_items=per_stream_items)

    def __len__(self):
        return self.effective_total_patches

class DatasetEpochCallback(TrainerCallback):
    def __init__(self, train_dataset=None, eval_dataset=None):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = 0 if state.epoch is None else int(state.epoch)

        if self.train_dataset is not None and hasattr(self.train_dataset, "set_epoch"):
            self.train_dataset.set_epoch(epoch)

        if self.eval_dataset is not None and hasattr(self.eval_dataset, "set_epoch"):
            self.eval_dataset.set_epoch(epoch)

        return control


def run_finetuning():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='diffugpt-s', choices=['diffugpt-s', 'diffugpt-m', 'diffullama'])
    parser.add_argument("--model_path", type=str, default="../Model")
    parser.add_argument("--base_model_name", type=str, default="gpt2", choices=['gpt', 'gpt-medium', 'sllama'])
    parser.add_argument("--data_dir", type=str, default="../Dataset/DIV2K/DIV2K_LR_unified/X4/train", help="Path to DIV2K")
    # parser.add_argument("--output_dir", type=str, default="./ddm-sft_output/diffugpt-s")
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    # parser.add_argument("--grad_acc_step", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epoch", type=int, default=30)
    
    args = parser.parse_args()
    
    args.grad_acc_step = max(1, 128 // args.batch_size)
    
    args.model_path = os.path.join(args.model_path, args.model_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subdir_name = f"train_full_{timestamp}"
    args.output_dir = os.path.join(args.model_path, "ddm-sft", subdir_name)
    
    # 获取当前进程的 local_rank (由 torchrun 自动注入环境变量)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank in [-1, 0]:
        wandb.init(project="ImageCompression-DiffuGPT", name=f"run_diffusteps{args.diffusion_steps}")
    
    # 1. Training Arguments
    print("Loading training_args and finetuning_args...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        per_device_train_batch_size=args.batch_size,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,  # 加速数据从内存到显存的传输
        gradient_accumulation_steps=args.grad_acc_step,
        learning_rate=args.lr,
        num_train_epochs=args.epoch,
        lr_scheduler_type='cosine',
        do_eval=True,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="steps",
        eval_steps=1000,
        metric_for_best_model="loss",   # 根据 loss 判断哪个模型最好
        load_best_model_at_end=True,    # 训练结束后加载验证集 loss 最低的模型
        # max_steps=69000,
        warmup_steps=2000,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=3,
        save_safetensors=False,  # 保存为 .bin 格式
        bf16=True,
        fp16=False,
        ddp_timeout=180000000,
        remove_unused_columns=False,  # 防止 input_ids 被过滤
        # resume_from_checkpoint="../Model/diffugpt-m/ddm-sft/train_full_20260115_133325/checkpoint-12000",
        # ignore_data_skip=True,  # 继续训练时忽略数据跳过
        report_to="wandb",
        # weight_decay=0.01,  # 增加权重衰减
        # ddp_find_unused_parameters=False,  # 关闭未用参数搜索，提升速度，消除警告
    )

    # 2. Finetuning Args (用于 Trainer 内部的 diffusion 参数)
    finetuning_args = FinetuningArguments(
        stage="ddm-sft",
        diffusion_steps=args.diffusion_steps,
        score_temp=1.0,
        logits_temp=1.0,
        anneal_steps=1,
        shift=True,
    )
    
    # 3. Load Model
    print("Loading tokenizer and DiscreteDiffusionModel...")
    tokenizer, model = load_ddm(args)
    # model.config.use_cache = False

    # 4. Prepare Dataset
    print("Loading DIV2K dataset...")
    train_dataset = Div2kPatchDataset(
        data_path=args.data_dir,
        tokenizer=tokenizer,
        num_chunks=-1,  # 根据dataset的__len__自动计算
        is_channel_wised=constants.IS_CHANNEL_WISED,
        shuffle=True,
        split='train',
        val_ratio=0.1,
        split_seed=42
    )
    eval_dataset = Div2kPatchDataset(
        data_path=args.data_dir,
        tokenizer=tokenizer,
        num_chunks=200,
        is_channel_wised=constants.IS_CHANNEL_WISED,
        shuffle=False,
        split='valid',
        val_ratio=0.1,
        split_seed=42
    )
    
    train_dataset.set_epoch(0)
    eval_dataset.set_epoch(0)
    
    # 5. Initialize Custom Trainer
    print("Loading ImageDiscreteDiffusionTrainer...")
    trainer = ImageDiscreteDiffusionTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=None,  # 图像patch大小固定，无需padding，默认collator即可
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        processor=None
    )
    
    trainer.add_callback(DatasetEpochCallback(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    ))

    print("Starting Training...")
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    print("Training Finished.")

if __name__ == "__main__":
    # NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_image_diffugpt.py
    # NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python train_image_diffugpt.py
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    run_finetuning()