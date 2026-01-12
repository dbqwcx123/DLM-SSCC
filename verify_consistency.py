import os
import torch
import torch.nn.functional as F
import argparse
import numpy as np
import matplotlib.pyplot as plt
from diffu_model import DiscreteDiffusionModel
from diffu_trainer import ImageDiscreteDiffusionTrainer
from transformers import AutoConfig, AutoTokenizer
from peft import PeftModel, LoraConfig
import imageio

from train_image_diffugpt import Div2kPatchDataset
import constants

def disable_dropout(model):
    """
    递归禁用模型中的 Dropout 层。
    这是验证数值精度的关键：必须排除随机性的干扰，只看浮点数误差。
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0

def calculate_consistency_metrics(model, dataloader, device, dtype_name):
    model.to(device)
    
    # 1. 彻底禁用 Dropout，排除结构性随机干扰
    disable_dropout(model)
    
    kl_divergences = []
    max_diffs = []
    
    print(f"\n开始验证 {dtype_name} 模式下的一致性 (Train vs Eval)...")
    
    # 只跑前 4 个 Batch (约 16 张图)
    for i, batch in enumerate(dataloader):
        # if i >= 4: break
        
        input_ids = batch["input_ids"].to(device)
        # attention_mask = batch["src_mask"].to(device) # DiffuGPT 内部会自动处理 mask
        
        # --- Pass 1: 推理模式 (Eval) ---
        model.eval()
        with torch.no_grad():
            # 获取 logits [B, Seq_Len, Vocab]
            # 注意：这里我们比较原始 Logits 即可，Shift 操作对两者是一样的
            logits_eval = model(input_ids, attention_mask=None)
            probs_eval = F.softmax(logits_eval, dim=-1) # [B, L, V]

        # --- Pass 2: 训练模式 (Train) ---
        model.train()
        # 不加 no_grad，模拟真实的 Training Forward
        logits_train = model(input_ids, attention_mask=None)
        probs_train = F.softmax(logits_train, dim=-1)

        # --- 计算差异 ---
        # 1. KL 散度: sum(p_train * log(p_train / p_eval))
        # F.kl_div 接收 log_prob 作为 input (第一个参数)，prob 作为 target (第二个参数)
        # 我们以 Eval 为基准 (Target)，Train 为近似 (Input)
        log_probs_train = F.log_softmax(logits_train, dim=-1)
        
        # reduction='batchmean' 会除以 batch_size
        kl = F.kl_div(log_probs_train, probs_eval, reduction='batchmean').item()
        
        # 2. 最大绝对误差 (Max Absolute Difference)
        diff = torch.abs(probs_train - probs_eval).max().item()
        
        kl_divergences.append(kl)
        max_diffs.append(diff)
        
        print(f"  Batch {i}: KL Div = {kl:.2e}, Max Diff = {diff:.2e}")

    avg_kl = np.mean(kl_divergences)
    avg_diff = np.mean(max_diffs)
    
    return avg_kl, avg_diff

def load_model_for_test(args, checkpoint_path, use_fp16):
    print(f"正在加载模型: {checkpoint_path} | 模式: {'FP16' if use_fp16 else 'BF16'}")
    
    # 1. 配置精度
    torch_dtype = torch.float16 if use_fp16 else torch.bfloat16
    
    # 2. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 3. 加载 Config
    config = AutoConfig.from_pretrained(args.base_model_path)
    
    # 4. 初始化基座模型 (GPT-2)
    model = DiscreteDiffusionModel.from_pretrained(
        args.base_model_path,
        model=args.base_model_name,
        config=config,
        tokenizer=tokenizer,
        device='cpu' # 先加载到 CPU，后面再转
    )
    
    # 5. 加载 LoRA 权重 (如果有)
    if os.path.exists(os.path.join(checkpoint_path, "adapter_model.bin")) or os.path.exists(os.path.join(checkpoint_path, "adapter_model.safetensors")):
        print("检测到 LoRA Adapter，正在加载...")
        model = PeftModel.from_pretrained(model, checkpoint_path)
    else:
        print("未检测到 LoRA Adapter")
        pass

    # 6. 转换精度并移动到 GPU
    model.to(dtype=torch_dtype).to("cuda")
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    # 你的 checkpoint 路径
    parser.add_argument("--bf16_checkpoint", type=str, default="./Model_test/test_bf16", help="BF16 训练出的模型路径")
    parser.add_argument("--fp16_checkpoint", type=str, default="./Model_test/test_fp16", help="FP16 训练出的模型路径")
    parser.add_argument("--base_model_path", type=str, default="../Model/diffugpt-s") # 或 diffugpt-m
    parser.add_argument("--base_model_name", type=str, default="gpt2")
    parser.add_argument("--data_dir", type=str, default="../Dataset/DIV2K/DIV2K_HR_test/train")
    args = parser.parse_args()

    # 准备数据 (只取前几个 Batch)
    # 临时加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    dataset = Div2kPatchDataset(
        data_path=args.data_dir,
        tokenizer=tokenizer,
        num_chunks=16, # 只需要少量数据
        is_channel_wised=False,
        shuffle=False
    )
    # 创建 DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)

    # --- 测试 BF16 模型 ---
    model_bf16, _ = load_model_for_test(args, args.bf16_checkpoint, use_fp16=False)
    bf16_kl, bf16_diff = calculate_consistency_metrics(model_bf16, dataloader, "cuda", "BF16")
    del model_bf16
    torch.cuda.empty_cache()

    # --- 测试 FP16 模型 ---
    model_fp16, _ = load_model_for_test(args, args.fp16_checkpoint, use_fp16=True)
    fp16_kl, fp16_diff = calculate_consistency_metrics(model_fp16, dataloader, "cuda", "FP16")
    del model_fp16
    torch.cuda.empty_cache()

    # --- 最终对比报告 ---
    print("\n" + "="*50)
    print("  CONSISTENCY CHECK REPORT (Training-Inference Mismatch)")
    print("="*50)
    print(f"{'Metric':<20} | {'BF16':<15} | {'FP16':<15} | {'Conclusion':<15}")
    print("-" * 70)
    print(f"{'Avg KL Divergence':<20} | {bf16_kl:.2e}        | {fp16_kl:.2e}        | {'FP16 Better' if fp16_kl < bf16_kl else 'BF16 Better'}")
    print(f"{'Max Abs Diff':<20} | {bf16_diff:.2e}        | {fp16_diff:.2e}        | {'FP16 Better' if fp16_diff < bf16_diff else 'BF16 Better'}")
    print("-" * 70)
    
    if fp16_kl < bf16_kl * 0.5: # 简单的阈值判断
        print("\n[结论]: 实验证实 FP16 显著减少了训练-推理的不匹配！")
        print("这对于压缩任务（算术编码）至关重要，因为推理时的概率必须与训练时高度一致。")
    else:
        print("\n[结论]: 两者差异不明显，可能需要更多训练步数或模型本身对此不敏感。")

if __name__ == "__main__":
    main()