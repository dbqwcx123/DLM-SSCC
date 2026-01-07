import argparse
import os
import struct
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import time
from PIL import Image

from diffu_model import *
import constants
from data_loaders import patch_visualize
from utils import arithmetic_coder
from utils.ac_utils import normalize_pdf_for_arithmetic_coding, bytes_to_bits
from utils.ECCT_utils import set_seed
from utils.pixel_token_dict import *

def decompress_image(bit_string, model, tokenizer, args):
    """
    使用DiffuGPT解压单个图像块
    输入: 单行比特流字符串
    输出: 解压后的 RGB 图像块 (H, W, C) numpy 数组
    """
    model.eval()
    device = model.device
    
    pixel_token_ids = compute_pixel_token_ids(tokenizer)
    
    # 1. 确定序列参数
    h, w = constants.CHUNK_SHAPE_2D
    channels = 1 if constants.IS_CHANNEL_WISED else 3
    seq_len = h * w * channels + 1  # 序列长度 = 像素数 + BOS
    batch_size = 1
    
    data_iter = iter(bit_string)
    
    # 2. 准备初始状态
    # 创建全掩码序列
    xt = torch.full((batch_size, seq_len), tokenizer.mask_token_id, dtype=torch.int64).to(device)
    maskable_mask = torch.ones((batch_size, seq_len), dtype=torch.bool).to(device)
    # 全双向注意力掩码
    attention_mask = get_anneal_attn_mask(seq_len, batch_size, dtype=torch.float32, device=device, attn_mask_ratio=1.0)
    
    # 设置 BOS Token
    xt[0, 0] = tokenizer.bos_token_id
    maskable_mask[0, 0] = False

    # 3. 初始化算术解码器
    def _input_fn(bit_sequence=data_iter):
        try:
            return int(next(bit_sequence))
        except StopIteration:
            return None
    
    decoder = arithmetic_coder.Decoder(
        base=constants.ARITHMETIC_CODER_BASE,
        precision=constants.ARITHMETIC_CODER_PRECISION,
        input_fn=_input_fn,
    )
    
    # 4. 扩散逆过程循环 (T-1 -> 0)
    for t in range(args.diffusion_steps-1, -1, -1):
        current_mask_indices = torch.nonzero(maskable_mask[0]).flatten()
        num_current_masks = len(current_mask_indices)
        
        if num_current_masks == 0:
            print(f"所有位置均已解码完成，提前退出循环。此时扩散步数 t={t}")
            break
            
        # 4.1 前向传播
        with torch.no_grad():
            raw_logits = model(xt, attention_mask=attention_mask)
        
        # Shift 操作
        logits_shifted = shift_logits(raw_logits)
        logits_pixel = logits_shifted[:,:, pixel_token_ids]
            
        # 4.2 计算置信度
        if args.confidence_st == 'entropy':
            confidences = get_confidence_entropy(logits_pixel, current_mask_indices)
        elif args.confidence_st == 'topk':
            confidences = get_confidence_topk(logits_pixel, current_mask_indices)
        elif args.confidence_st == 'simple':
            confidences = get_confidence_simple(logits_pixel, current_mask_indices)
        else:
            raise ValueError(f"未知的置信度计算方法: {args.confidence_st}")
        
        # 4.3 排序
        # sorted_indices = torch.argsort(confidences, descending=True)
        # sorted_mask_pos = current_mask_indices[sorted_indices]
        sorted_mask_pos = conf_based_sorting(confidences, current_mask_indices, device)
        
        # 4.4 确定本步解码数量
        ratio = 1.0 / (t + 1)
        k = int(num_current_masks * ratio)
        k = max(1, k)
        k = min(k, num_current_masks)
        
        target_indices = sorted_mask_pos[:k]
        
        if args.verbose:
             # 获取当前最高置信度用于监控
            print(f"Step {t}: 剩余 {num_current_masks}, 解码 {k} 个位置，比例 {ratio:.4f}")
        
        # 4.5 逐个解码
        for idx in target_indices:
            idx = idx.item()
            
            # prob_dist = torch.softmax(logits_pixel[0, idx], dim=-1)
            # prob_dist = prob_dist.cpu().numpy()
            
            # 移动到 CPU 并转为 double 计算 Softmax，获取预测的概率分布
            logits_fp64 = logits_pixel[0, idx].detach().cpu().double()
            prob_dist = torch.softmax(logits_fp64, dim=-1)
            # prob_dist = smooth_probs(prob_dist, k=args.smooth_k, alpha=args.smooth_alpha)
            prob_dist = prob_dist.numpy()
            
            try:
                decoded_pixel_value = decoder.decode(normalize_pdf_for_arithmetic_coding(prob_dist))
                if not (0 <= decoded_pixel_value <= 255):
                    print(f"错误: 解码值 {decoded_pixel_value} 超出pixel_token_ids范围")
                    decoded_pixel_value = 128  # 沿用D3PM设置，解码失败则为灰像素
            except Exception as e:
                # print(f"解码错误 at step {t}, idx {idx}: {e}")
                decoded_pixel_value = 128  # 沿用D3PM设置，解码失败则为灰像素
            
            decoded_pixel_id = pixel_token_ids[decoded_pixel_value]
            xt[0, idx] = decoded_pixel_id
            maskable_mask[0, idx] = False
            
    # 5. 重建像素矩阵
    output_ids = xt[0].tolist()
    
    # 去掉 BOS
    if output_ids[0] == tokenizer.bos_token_id:
        output_ids = output_ids[1:]
        
    # Token ID -> String -> Int
    token_strs = tokenizer.convert_ids_to_tokens(output_ids)
    
    pixel_values = []
    for s in token_strs:
        # 处理可能的特殊字符（如 GPT2 的 Ġ）
        s_clean = s.replace('Ġ', '')
        try:
            val = int(s_clean)
            if 0 <= val <= 255:
                pixel_values.append(val)
            else:
                pixel_values.append(128)  # 沿用D3PM设置，解码失败则为灰像素
        except ValueError:
            pixel_values.append(128)  # 沿用D3PM设置，解码失败则为灰像素
            
    # 转换为 Numpy 数组
    # reconstructed_patch = np.array(pixel_values, dtype=np.uint8)
    reconstructed_patch = np.array(pixel_values).astype(np.uint8)
    
    # 形状检查与重塑
    expected_len = h * w * channels
    if len(reconstructed_patch) != expected_len:
        print(f"警告: 解码长度 {len(reconstructed_patch)} 不匹配，预期 {expected_len}")
        if len(reconstructed_patch) > expected_len:
            reconstructed_patch = reconstructed_patch[:expected_len]
        else:
            reconstructed_patch = np.pad(reconstructed_patch, (0, expected_len - len(reconstructed_patch)), 'constant')
    
    reconstructed_patch = reconstructed_patch.reshape(h, w, channels)
    
    return reconstructed_patch

def reconstruct_image_from_patches(patches, image_shape):
    """
    将多个 patch 拼合成完整图像
    假设 patch 是按行优先顺序排列的
    """
    height, width, channels = image_shape
    h_patch, w_patch = constants.CHUNK_SHAPE_2D
    
    full_image = np.zeros((height, width, channels), dtype=np.uint8)
    
    cols = width // w_patch
    
    for idx, patch in enumerate(patches):
        row = idx // cols
        col = idx % cols
        
        y_start = row * h_patch
        y_end = y_start + h_patch
        x_start = col * w_patch
        x_end = x_start + w_patch
        
        # 确保不越界
        if y_end <= height and x_end <= width:
             full_image[y_start:y_end, x_start:x_end] = patch
             
    return full_image

def main(args):
    print(f"模型路径: {args.model_path}")
    print(f"基础模型: {args.base_model_name}")
    
    # 1. 加载模型和分词器
    tokenizer, model = load_ddm(args)
    patches_per_image = constants.PATCHES_PER_IMAGE
    
    # 2. 准备输入输出
    if not os.path.exists(args.input_dir):
        print(f"错误: 文件 {args.input_dir} 不存在，请先运行压缩脚本。")
        return

    ## channel = AWGN/Rayleigh
    for SNR in range(6, -4, -1):
        print(f"\n{'='*30}\n开始解压 SNR={SNR} 的文件...")
        args.input_file = args.input_dir + '/demo_decode_SNR_' + str(SNR) + '.txt'
        save_dir_reconstructed = os.path.join(args.output_dir, f'SNR_{SNR}')
        os.makedirs(save_dir_reconstructed, exist_ok=True)
    ## channel = None
    # for SNR in [0]:
    #     args.input_file = args.input_dir + '/compressed_output.txt'
    #     save_dir_reconstructed = args.output_dir
        
        print(f"正在读取压缩文件: {args.input_file}")
        with open(args.input_file, 'r') as f:
            lines = f.readlines()
        
        current_patches = []
        image_idx = 0
        
        total_start_time = time.time()
        
        # 3. 逐行解压 (每一行是一个 patch 的比特流)
        print(f"开始解压，共 {len(lines)} 个 Patch...")
        for line_idx, line in enumerate(tqdm(lines)):
            raw_bit_string = line.strip()
            if not raw_bit_string:
                continue
            
            ###移除停止位
            # 1. 去除末尾所有的 '0' (Padding)
            stripped_padding = raw_bit_string.rstrip('0')
            
            # 2. 去除末尾的停止位 '1'
            if len(stripped_padding) > 0:
                bit_string = stripped_padding[:-1]
            else:
                # 这种情况理论上不应该发生，除非信道误码把停止位 '1' 变成了 '0'
                # 或者原始数据本身就是空的
                print(f"警告: 第 {line_idx} 行数据异常，未找到停止位")
                bit_string = ""
                
            # 3. 解压单个 Patch
            patch = decompress_image(bit_string, model, tokenizer, args)
            current_patches.append(patch)
            
            # 当收集满一张图的所有 Patch 时进行重组
            if len(current_patches) == patches_per_image:
                print(f"\n正在重组第 {image_idx} 张图像...")
                
                # 使用 constants 中定义的完整图像尺寸，例如 CIFAR10: (32, 32, 3)
                full_image = reconstruct_image_from_patches(current_patches, constants.IMAGE_SHAPE)
                
                # 保存完整图像
                img_save_path = os.path.join(save_dir_reconstructed, f'image_{image_idx}.png')
                Image.fromarray(full_image).save(img_save_path)
                print(f"图像已保存至: {img_save_path}")
                
                current_patches = []
                image_idx += 1
                
        total_end_time = time.time()
        print("="*30)
        print(f"解压完成，总耗时: {total_end_time - total_start_time:.2f}s")
        print(f"解压结果保存在: {save_dir_reconstructed}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='diffugpt-s', choices=['diffugpt-s', 'diffugpt-m', 'diffullama'])
    parser.add_argument("--base_model_name", type=str, default='gpt2', choices=['gpt2', 'gpt2-medium', 'llama'])
    parser.add_argument("--model_path", type=str, default='../Model', help="DiffuGPT path")
    parser.add_argument("--ddm_sft", type=bool, default=True, help="是否使用微调后的DiffuGPT模型")
    parser.add_argument("--checkpoint_dir", type=str, default='train_20251228_192149')
    parser.add_argument("--checkpoint_name", type=str, default='checkpoint-48000')
    parser.add_argument("--diffusion_steps", type=int, default=50)
    parser.add_argument("--confidence_st", type=str, default='entropy', choices=['entropy', 'topk', 'simple'], help="置信度计算策略")
    parser.add_argument('--verbose', type=bool, default=False, help='打印详细过程')
    parser.add_argument("--channel", type=str, default='AWGN', choices=[None, 'AWGN', 'Rayleigh'])
    parser.add_argument("--dataset_type", type=str, default="DIV2K")
    parser.add_argument("--root_dir", type=str, default="./image_io", help="输入输出文件夹的根目录")
    
    args = parser.parse_args()
    
    args.model_path = os.path.join(args.model_path, args.model_name)
    if args.ddm_sft:
        args.model_path = os.path.join(args.model_path, "ddm-sft", args.checkpoint_dir, args.checkpoint_name)
    args.root_dir = os.path.join(args.root_dir, f'{args.dataset_type}', f'{args.confidence_st}_confidence', f'smooth_k{args.smooth_k}_alpha{args.smooth_alpha}')
    args.root_dir = os.path.join(args.root_dir, 'channel_indep') if constants.IS_CHANNEL_WISED else os.path.join(args.root_dir, 'channel_corre')
    if args.ddm_sft:
        args.root_dir = os.path.join(args.root_dir, f'patch{constants.CHUNK_SHAPE_2D}', f'{args.model_name}_ddm-sft', f'{args.checkpoint_dir}', f'diffu_step{args.diffusion_steps}')
    else:
        args.root_dir = os.path.join(args.outpuroot_dirt_path, f'patch{constants.CHUNK_SHAPE_2D}', f'{args.model_name}', f'diffu_step{args.diffusion_steps}')
    
    if args.channel:
        args.input_dir  = os.path.join(args.root_dir, f"ECCT_forward/LDPC_K24_N49/{args.channel}")
        args.output_dir = os.path.join(args.root_dir, f"ECCT_reconstruct/LDPC_K24_N49/{args.channel}")
    else:
        args.input_dir  = args.root_dir
        args.output_dir = args.root_dir
    
    return args

if __name__ == "__main__":
    set_seed(42)
    main(get_args())