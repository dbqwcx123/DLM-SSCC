import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 关闭 tokenizers 的并行以避免警告
import sys
import argparse
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, TrainingArguments
from torch.utils.data import IterableDataset, get_worker_info
import torch.distributed as dist
import math
import imageio
from natsort import natsorted
import wandb
from datetime import datetime
import random

from peft import LoraConfig, get_peft_model, TaskType, PeftModel

import constants
from diffu_model import *
from diffu_trainer import ImageDiscreteDiffusionTrainer
from llamafactory.hparams import FinetuningArguments



class Div2kPatchDataset(IterableDataset):
    def __init__(self, data_path, tokenizer, num_chunks=-1, is_channel_wised=False, shuffle=True, split='train'):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.num_chunks = num_chunks
        self.is_channel_wised = is_channel_wised
        self.shuffle = shuffle
        self.split = split
        
        # 1. 预先加载所有文件路径 (移出 __iter__ 以便切分)
        if not os.path.exists(data_path):
            raise ValueError(f"Data path {data_path} does not exist.")
        
        self.all_files = [
            os.path.join(data_path, item) 
            for item in os.listdir(data_path) 
            if item.lower().endswith('.png')
        ]
        self.all_files = natsorted(self.all_files)
        if shuffle:
            # 设定固定的随机种子，保证每次实验打乱顺序一致，且所有 worker 看到的乱序是一样的
            random.seed(42) 
            random.shuffle(self.all_files)
        print(f"Dataset initialized: Found {len(self.all_files)} images.")

    def process_patch_to_tokens(self, patch):
        flat_pixels = patch.flatten() 
        num_str_tokens = [str(val) for val in flat_pixels]
        input_ids = [self.tokenizer.bos_token_id] + self.tokenizer.convert_tokens_to_ids(num_str_tokens)
        return torch.tensor(input_ids, dtype=torch.long)

    def _get_image_iterator(self, files):
        """内部生成器：处理指定的文件列表"""
        h, w = constants.CHUNK_SHAPE_2D
        idx = 0
        
        for file_path in files:
            # 读取图像
            image = imageio.v2.imread(file_path)
            height, width = image.shape[0], image.shape[1]
            
            # 生成 patches
            # 将 _extract_image_patches 内联，以便处理多进程逻辑
            patches = []
            if self.is_channel_wised:
                 for i in range(image.shape[-1]):
                    temp_data = image[:, :, i:i+1]
                    # 简单的切片循环
                    for row in range(height // h):
                        for col in range(width // w):
                            patches.append(temp_data[row * h: (row + 1) * h, col * w: (col + 1) * w])
            else:
                for row in range(height // h):
                    for col in range(width // w):
                        patches.append(image[row * h: (row + 1) * h, col * w: (col + 1) * w])

            # Yield 处理
            for patch in patches:
                if self.num_chunks > 0 and idx >= self.num_chunks:
                    return
                
                # 处理成 Trainer 需要的格式
                input_ids = self.process_patch_to_tokens(patch)
                src_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                if input_ids[0] == self.tokenizer.bos_token_id:
                    src_mask[0] = True
                
                yield {
                    "input_ids": input_ids,
                    "src_mask": src_mask
                }
                idx += 1

    def __iter__(self):
        # --- 处理多卡 DDP 环境下的数据切分 ---
        if dist.is_initialized():
            num_gpus = dist.get_world_size()
            gpu_id = dist.get_rank()
        else:
            num_gpus = 1
            gpu_id = 0

        # 先把总文件列表按 GPU 数量均分
        per_gpu_files_count = int(math.ceil(len(self.all_files) / float(num_gpus)))
        start_gpu = gpu_id * per_gpu_files_count
        end_gpu = min(start_gpu + per_gpu_files_count, len(self.all_files))
        files_on_this_gpu = self.all_files[start_gpu:end_gpu]
        
        # --- 处理单卡内的多 Worker 切分 ---
        worker_info = get_worker_info()
        
        if worker_info is None:
            # 单进程模式
            files_to_process = files_on_this_gpu
        else:
            # 多进程模式：基于当前 GPU 分到的文件，再分给每个 worker
            per_worker = int(math.ceil(len(files_on_this_gpu) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(files_on_this_gpu))
            files_to_process = files_on_this_gpu[start:end]
            
        # 调用内部生成器
        return self._get_image_iterator(files_to_process)

    def __len__(self):
        if self.num_chunks > 0:
            return self.num_chunks
        else:
            if self.split == 'train':
                return constants.NUM_CHUNKS_TRAIN
            elif self.split == 'valid':
                return constants.NUM_CHUNKS_VALID
            else:
                return constants.NUM_CHUNKS_TEST

def run_finetuning():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='diffugpt-s', choices=['diffugpt-s', 'diffugpt-m', 'diffullama'])
    parser.add_argument("--model_path", type=str, default="../Model")
    parser.add_argument("--base_model_name", type=str, default="gpt2", choices=['gpt', 'gpt-medium', 'sllama'])
    parser.add_argument("--data_dir", type=str, default="../Dataset/DIV2K/DIV2K_HR_test", help="Path to DIV2K")
    # parser.add_argument("--output_dir", type=str, default="./ddm-sft_output/diffugpt-s")
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    # parser.add_argument("--grad_acc_step", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)  # LoRA 训练可以用大一点的学习率
    parser.add_argument("--epoch", type=int, default=5)

    parser.add_argument("--use_lora", type=bool, default=True)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    args = parser.parse_args()
    
    args.grad_acc_step = max(1, 128 // args.batch_size)
    
    args.model_path = os.path.join(args.model_path, args.model_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subdir_name = f"train_lora_{timestamp}" if args.use_lora else f"train_full_{timestamp}"
    args.output_dir = os.path.join(args.model_path, "ddm-sft", subdir_name)
    args.output_dir = './Model_test/test_bf16'
    
    # 获取当前进程的 local_rank (由 torchrun 自动注入环境变量)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank in [-1, 0]:
        wandb.init(project="ImageCompression-DiffuGPT", name=f"run_diffusteps{args.diffusion_steps}")
    # 1. Load Model
    print("Loading tokenizer and DiscreteDiffusionModel...")
    tokenizer, model = load_ddm(args)
    
    if args.use_lora:
        print(f"Applying LoRA (Rank={args.lora_rank}, Alpha={args.lora_alpha})...")
        # GPT-2 的注意力模块通常命名为 c_attn
        peft_config = LoraConfig(
            inference_mode=False, 
            r=args.lora_rank, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=0.1,
            target_modules=["c_attn"] # 针对 GPT-2 架构
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters() # 打印可训练参数量，确认 LoRA 生效
    
    # 2. Prepare Dataset
    print("Loading DIV2K dataset...")
    train_dataset = Div2kPatchDataset(
        data_path=args.data_dir+"/train",
        tokenizer=tokenizer,
        num_chunks=-1,  # 根据dataset的__len__自动计算
        is_channel_wised=constants.IS_CHANNEL_WISED,
        shuffle=True,
        split='train'
    )
    eval_dataset = Div2kPatchDataset(
        data_path=args.data_dir+"/valid",
        tokenizer=tokenizer,
        num_chunks=2000,
        is_channel_wised=constants.IS_CHANNEL_WISED,
        shuffle=True,
        split='valid'
    )
    
    # 3. Training Arguments
    print("Loading training_args and finetuning_args...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        per_device_train_batch_size=args.batch_size,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,  # 加速数据从内存到显存的传输
        gradient_accumulation_steps=args.grad_acc_step,
        learning_rate=args.lr,
        num_train_epochs=args.epoch,
        lr_scheduler_type='cosine',
        do_eval=True,
        per_device_eval_batch_size=args.batch_size*8,
        evaluation_strategy="steps",
        eval_steps=2000,
        metric_for_best_model="loss",   # 根据 loss 判断哪个模型最好
        load_best_model_at_end=True,    # 训练结束后加载验证集 loss 最低的模型
        max_steps=100,
        warmup_steps=10,
        logging_steps=5,
        save_steps=2000,
        save_total_limit=3,
        save_safetensors=False,  # 保存为 .bin 格式
        bf16=True,
        fp16=False,
        # deepspeed='ds_z2_config.json',
        ddp_timeout=180000000,
        remove_unused_columns=False,  # 重要：防止 input_ids 被过滤
        # resume_from_checkpoint="../Model/diffugpt-m/ddm-sft/train_20260103_162805/checkpoint-38000",
        # ignore_data_skip=True,  # 继续训练时忽略数据跳过
        report_to="wandb",
        weight_decay=0.01,  # 增加权重衰减
        # LoRA 训练时，只保存 Adapter 可以节省空间
        save_only_model=True if args.use_lora else False
    )
    
    # 4. Finetuning Args (用于 Trainer 内部的 diffusion 参数)
    finetuning_args = FinetuningArguments(
        stage="ddm-sft",
        diffusion_steps=args.diffusion_steps,
        score_temp=1.0,
        logits_temp=1.0,
        anneal_steps=1,
        shift=True,
    )
    
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

    print("Starting Training...")
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    print("Training Finished.")

if __name__ == "__main__":
    # NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_diffugpt_test.py
    # NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python train_diffugpt_test.py
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    run_finetuning()