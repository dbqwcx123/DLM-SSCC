import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import argparse

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from train_image_diffugpt import Div2kPatchDataset
from diffu_model import load_ddm

def verify_dataset(dataset):
    print("正在获取第一个样本进行验证...")
    sample = dataset[0] # 获取第一个样本
    input_ids = sample['input_ids'].tolist()
    
    # 1. 去掉 BOS Token (假设第一个是 BOS)
    # 注意：你需要确认 dataset.bos_token_id 是多少，通常检查 input_ids[0]
    if input_ids[0] == dataset.bos_token_id:
        token_ids = input_ids[1:] 
    
    # 2. Token IDs -> String -> Int
    # 这一步依赖于 tokenizer 的具体实现，假设它是将 "128" 编码为 token
    try:
        decoded_str_list = dataset.tokenizer.convert_ids_to_tokens(token_ids)
        decoded_text = dataset.tokenizer.decode(token_ids)
        # tokenizer decode 出来是空格分隔的字符串 "123 45 0 255 ..."
        pixel_vals = [int(p) for p in decoded_text.strip().split()]
    except Exception as e:
        print(f"解码失败，尝试直接转换 (依赖 Tokenizer 类型): {e}")
        # 备用方案：如果 tokenizer 是简单的字符映射
        pixel_vals = []
        for tid in token_ids:
            s = dataset.tokenizer.convert_ids_to_tokens(tid)
            # 清洗特殊字符，例如 RoBERTa/GPT2 的 'Ġ'
            s = s.replace('Ġ', '') 
            pixel_vals.append(int(s))

    # 3. 检查数据长度
    h, w = dataset.patch_size
    expected_len = h * w * 3
    if len(pixel_vals) != expected_len:
        print(f"错误：解码后的像素数量 {len(pixel_vals)} 与预期 {expected_len} 不符！")
        return

    # 4. Reshape 回图像 (H, W, C)
    # 注意：这里必须与你 Dataset 中的 flatten 顺序对应
    # 你的代码使用的是 patch.flatten()，这是行优先：Row1, Row2...
    img_array = np.array(pixel_vals, dtype=np.uint8).reshape(h, w, 3)

    # 5. 可视化对比
    plt.figure(figsize=(4, 4))
    plt.imshow(img_array)
    plt.title(f"Reconstructed Patch {h}x{w}")
    plt.axis('off')
    plt.show()
    print("验证完成，请检查弹出的图像是否为正常的图像块（非花屏）。")

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='diffugpt-s', choices=['diffugpt-s', 'diffugpt-m'])
parser.add_argument("--base_model_name", type=str, default='gpt2', choices=['gpt2', 'gpt2-medium'])
parser.add_argument("--model_path", type=str, default='../Model')
parser.add_argument("--diffusion_steps", type=int, default=100)
parser.add_argument("--confidence_st", type=str, default='entropy', choices=['entropy', 'topk', 'simple'], help="置信度计算策略")
parser.add_argument("--smooth_k", type=int, default=1, help="概率平滑半径")
parser.add_argument("--smooth_alpha", type=float, default=0.1, help="概率平滑强度")
parser.add_argument('--verbose', type=bool, default=False, help='打印详细过程')
parser.add_argument("--keep_bos", type=bool, default=True, help="是否保留BOS不压缩(作为已知条件)")
parser.add_argument("--data_dir", type=str, default="../Dataset/DIV2K/DIV2K_train_LR/X4_test", help="Path to DIV2K")
parser.add_argument("--output_path", type=str, default="./image_io")
args = parser.parse_args()

args.model_path = os.path.join(args.model_path, args.model_name)

tokenizer, model = load_ddm(args)
verify_dataset(Div2kPatchDataset(args.data_dir, tokenizer, patch_size=(16, 16)))