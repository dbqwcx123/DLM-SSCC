import argparse
import os
import struct
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import time

from transformers import AutoConfig, AutoTokenizer
from diffu_model import *

import constants, data_loaders
from image_compress_diffugpt import compress_image

def main(args):    
    idx = 0
    total_bits = 0
    h, w = constants.CHUNK_SHAPE_2D
    channels = 1 if constants.IS_CHANNEL_WISED else 3  # 图像块的通道数
    
    # 1. 准备数据
    ##TODO: 数据加载RGB顺序要改一下
    data_iterator = data_loaders.get_cifar10_iterator(
                    num_chunks=constants.NUM_CHUNKS,
                    is_channel_wised=constants.IS_CHANNEL_WISED,
                    is_seq=False,  # 按图像块顺序提取
                    data_path=args.input_path)
    
    print(f"开始扩散压缩过程，总步数 T={args.diffusion_steps}...")
    for data, frame_id in tqdm(data_iterator):
        # frame_id是图像编号，idx是patch编号
        # if frame_id in [3,6,25,0,22,12,4,13,1,11]:  # 每个类别取1张图，共10张
        if frame_id in [0]:  # 测试用
            # print(f"处理序号为 {frame_id} 的图片的第 {idx % constants.PATCHES_PER_IMAGE + 1} 个图像块...")
            seq_array = data
            seq_array = seq_array.reshape(1, h*w*channels)  # [1, h*w*c]，同时验证大小是否正确
            flattened_array = seq_array.flatten()  # 展平为一维数组
            num_str_tokens = [str(num) for num in flattened_array]  # 数字转换为字符串
            
            compressed_bits, seq_len = compress_image(num_str_tokens, model, tokenizer, args)
            
            num_bits = len(compressed_bits)
            total_bits += num_bits
            
        idx += 1
    print(f"\n所有图像块总压缩比特数: {total_bits}")
    return total_bits
    
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='../Model/diffugpt-s')
    parser.add_argument("--base_model_name", type=str, default='gpt2')
    # parser.add_argument("--diffusion_steps", type=int, default=110)
    parser.add_argument("--confidence_st", type=str, default='entropy', choices=['entropy', 'topk', 'simple'], help="置信度计算策略")
    parser.add_argument("--smooth_k", type=int, default=0, help="概率平滑半径")
    parser.add_argument("--smooth_alpha", type=float, default=0, help="概率平滑强度")
    parser.add_argument('--verbose', type=bool, default=False, help='打印详细过程')
    parser.add_argument("--keep_bos", type=bool, default=True, help="是否保留BOS不压缩(作为已知条件)")
    parser.add_argument("--input_path", type=str, default="../Dataset/CIFAR10")
    # parser.add_argument("--output_path", type=str, default="./image_io/diffugpt-s/diffu_step_max")
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = get_args()
    
    print(f"模型路径: {args.model_path}")
    print(f"基础模型: {args.base_model_name}")
    
    # 加载模型
    tokenizer, model = load_ddm(args)
    
    steps_list = list(range(80, 160, 10))
    results = []
    for s in tqdm(steps_list):
        args.diffusion_steps = s
        total_bits = main(args)
        results.append((s, total_bits))
    # 保存为 CSV
    import csv
    csv_path ="./diffugpt-s_steps_vs_bits.csv"
    with open(csv_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["diffusion_steps", "total_bits"])
        writer.writerows(results)

    # 绘图并保存为 PNG（使用无头后端）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs, ys = zip(*results)
    plt.figure(figsize=(8,5))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("diffusion_steps")
    plt.ylabel("total_bits")
    plt.title("total_bits vs diffusion_steps")
    plt.grid(True)
    fig_path = "./diffugpt-s_steps_vs_bits.png"
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    print(f"CSV: {csv_path}")
    print(f"图像: {fig_path}")