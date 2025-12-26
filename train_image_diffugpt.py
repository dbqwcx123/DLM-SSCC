import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 关闭 tokenizers 的并行以避免警告
import sys
import argparse
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, TrainingArguments
from torch.utils.data import IterableDataset, get_worker_info
import math
import imageio
from natsort import natsorted
import wandb
from datetime import datetime
import random

from data_loaders import get_div2k_iterator
import constants
from diffu_model import *
from diffu_trainer import ImageDiscreteDiffusionTrainer
from llamafactory.hparams import FinetuningArguments



class Div2kPatchDataset(IterableDataset):
    def __init__(self, data_path, tokenizer, num_chunks=-1, is_channel_wised=False, shuffle=True):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.num_chunks = num_chunks
        self.is_channel_wised = is_channel_wised
        self.shuffle = shuffle
        
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
            
            # 生成 patch 的逻辑
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
                
                # 在此处处理成 Trainer 需要的格式
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
        # 2. 关键修改：多进程切分逻辑
        worker_info = get_worker_info()
        
        if worker_info is None:
            # 单进程模式 (num_workers=0)
            files_to_process = self.all_files
        else:
            # 多进程模式：将文件列表平均分给每个 worker
            per_worker = int(math.ceil(len(self.all_files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.all_files))
            files_to_process = self.all_files[start:end]
            
        # 调用内部生成器
        return self._get_image_iterator(files_to_process)

    def __len__(self):
        return constants.NUM_CHUNKS
    
    

def run_finetuning():
    # "../Dataset/DIV2K/DIV2K_HR_unified/train"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../Model/diffugpt-s", help="DiffuGPT path or base model")
    parser.add_argument("--base_model_name", type=str, default="gpt2")
    parser.add_argument("--data_dir", type=str, default="../Dataset/DIV2K/DIV2K_HR_unified/train", help="Path to DIV2K")
    # parser.add_argument("--output_dir", type=str, default="./ddm-sft_output/diffugpt-s")
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_acc_step", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epoch", type=int, default=2)
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.model_path, "ddm-sft", f"train_{timestamp}")
    
    wandb.init(project="ImageCompression-DiffuGPT", name=f"run_diffusteps{args.diffusion_steps}")
    # 1. Load Model (Using User's Logic)
    print("Loading tokenizer and DiscreteDiffusionModel...")
    # 注意：这里使用了你提供的 model.py 中的类
    tokenizer, model = load_ddm(args)

    # 2. Prepare Dataset
    print("Loading DIV2K dataset...")
    dataset = Div2kPatchDataset(
        data_path=args.data_dir,
        tokenizer=tokenizer,
        num_chunks=-1,  # 读取全部数据
        is_channel_wised=constants.IS_CHANNEL_WISED,
        shuffle=True
    )
    
    # 3. Training Arguments
    print("Loading training_args and finetuning_args...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        per_device_train_batch_size=args.batch_size,
        dataloader_num_workers=20,
        dataloader_pin_memory=True,  # 加速数据从内存到显存的传输
        gradient_accumulation_steps=args.grad_acc_step,
        learning_rate=args.lr,
        num_train_epochs=args.epoch,
        lr_scheduler_type='cosine',
        max_steps=10000,
        warmup_steps=1000,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        save_safetensors=False,  # 保存为 .bin 格式
        bf16=True,
        # fp16=True if torch.cuda.is_available() else False,
        # deepspeed='ds_z2_config.json',
        ddp_timeout=180000000,
        remove_unused_columns=False,  # 重要：防止 input_ids 被过滤
        report_to="wandb",
    )

    # 4. Finetuning Args (用于 Trainer 内部的 diffusion 参数)
    finetuning_args = FinetuningArguments(
        stage="ddm-sft",
        diffusion_steps=args.diffusion_steps,
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
        train_dataset=dataset,
        # eval_dataset=dataset,
        tokenizer=tokenizer,
        processor=None
    )

    print("Starting Training...")
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    model.config.save_pretrained(args.output_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    print("Training Finished.")

if __name__ == "__main__":
    run_finetuning()