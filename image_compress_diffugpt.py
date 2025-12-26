import argparse
import os
import struct
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import time

from diffu_model import *
import constants, data_loaders
from utils import arithmetic_coder
from utils.ac_utils import normalize_pdf_for_arithmetic_coding, bits_to_bytes
from utils.ECCT_utils import set_seed
from utils.pixel_token_dict import *


def compress_image(num_str_tokens, model, tokenizer, args):
    """
    使用DiffuGPT压缩图像块的像素值序列
    """
    model.eval()
    device = model.device
    
    pixel_token_ids = compute_pixel_token_ids(tokenizer)
    
    # 添加 BOS token (DiffuGPT 训练习惯)，将数字字符串转换为token_id序列
    input_ids = [tokenizer.bos_token_id] + tokenizer.convert_tokens_to_ids(num_str_tokens)
    #######################################################################
    
    x = torch.tensor([input_ids]).to(device) # [1, seq_len]
    
    x_embed = model.get_embeds(x)
    seq_len = x.size(1)
    batch_size = x.size(0)
    
    # 2. 创建初始掩码状态
    # 创建全 mask 的序列作为起点
    src_mask = torch.zeros_like(x, dtype=torch.bool).to(device)
    maskable_mask = ~src_mask
    # 完全双向注意力掩码，attn_mask_ratio=1.0 表示全0掩码
    attention_mask = get_anneal_attn_mask(seq_len, batch_size, dtype=x_embed.dtype, device=device, attn_mask_ratio=1.0)
    
    # 如果保留 BOS 不压缩（作为 Prompt），则将第一个位置设为 False
    if args.keep_bos:
        maskable_mask[0, 0] = False
    # 初始状态：所有位置都是[MASK]
    xt = x.masked_fill(maskable_mask, tokenizer.mask_token_id)
    
    # 3. 初始化算术编码器
    output_bits = []
    encoder = arithmetic_coder.Encoder(
        base=constants.ARITHMETIC_CODER_BASE,
        precision=constants.ARITHMETIC_CODER_PRECISION,
        output_fn=output_bits.append,
    )
    
    # 4. 扩散逆过程循环
    # 使用从 T-1 到 0 的循环。
    # 逻辑说明：
    # t = T-1 (第一步): 对应原代码的 t=T。此时上下文最少，预测最难。
    # ...
    # t = 0 (最后一步): 对应原代码的最后收尾。此时上下文最多，预测最准。
    # 注意：原代码是 range(steps-1, 0, -1)，这里改为包含0以确保收敛
    for t in range(args.diffusion_steps-1, -1, -1):
        current_mask_indices = torch.nonzero(maskable_mask[0]).flatten()
        num_current_masks = len(current_mask_indices)
        
        if num_current_masks == 0:
            break
            
        # 4.1 前向传播预测概率 (Non-Autoregressive Step)
        # 这一步计算出的 logits 将用于本步内所有被选中 token 的编码
        with torch.no_grad():
            raw_logits = model(xt, attention_mask=attention_mask)
            # logits: [1, seq_len, vocab_size]
            
        # 经过移位操作后，logits_shifted[:, i] 才是针对 xt[:, i] 的预测分布
        logits_shifted = shift_logits(raw_logits)
        logits_pixel = logits_shifted[:,:, pixel_token_ids]
        
        # 4.2 计算置信度并决定去噪顺序
        if args.confidence_st == 'entropy':
            confidences = get_confidence_entropy(logits_pixel, current_mask_indices)
        elif args.confidence_st == 'topk':
            confidences = get_confidence_topk(logits_pixel, current_mask_indices)
        elif args.confidence_st == 'simple':
            confidences = get_confidence_simple(logits_pixel, current_mask_indices)
        else:
            raise ValueError(f"未知的置信度计算方法: {args.confidence_st}")
        
        # 4.3 确定本步处理顺序 (根据置信度降序，稳定排序策略)
        # sorted_indices = torch.argsort(confidences, descending=True)
        # sorted_mask_pos = current_mask_indices[sorted_indices]
        sorted_mask_pos = conf_based_sorting(confidences, current_mask_indices, device)
        
        # 4.4 确定本步解压数量 k
        # 策略：按照 1/(t+1) 的比例选择。
        # 当 t=T-1 时，比例是 1/T。
        # 当 t=0 时，比例是 1/1 = 100% (清理剩余所有)。
        ratio = 1.0 / (t + 1)
        k = int(num_current_masks * ratio)
        # 防止因概率计算导致 k=0 的空转
        k = max(1, k)
        k = min(k, num_current_masks) # 确保不越界
        
        # 选出本步要处理的 k 个位置
        target_indices = sorted_mask_pos[:k]
        
        if args.verbose:
            print(f"Step {t}: 剩余 {num_current_masks}, 去噪 {k} 个位置，比例 {ratio:.4f}")
        
        # 4.5 逐个编码 (Step-wise Compression)
        # 这里的关键是：我们使用步骤 4.1 计算出的同一个 logits。
        # 这意味着这 k 个 token 是“并行”预测的，但“串行”编码写入比特流。
        for idx in target_indices:
            idx = idx.item()
            # 获取该位置的真实 token_id
            true_pixel_id = x[0, idx].item()
            true_pixel_value = tokenid_to_pixel(true_pixel_id, tokenizer)
            
            # prob_dist = torch.softmax(logits_pixel[0, idx], dim=-1)
            # prob_dist = smooth_probs(prob_dist, k=args.smooth_k, alpha=args.smooth_alpha)
            # prob_dist = prob_dist.cpu().numpy()
            
            # 移动到 CPU 并转为 double 计算 Softmax，获取预测的概率分布
            logits_fp64 = logits_pixel[0, idx].detach().cpu().double()
            prob_dist = torch.softmax(logits_fp64, dim=-1)
            prob_dist = smooth_probs(prob_dist, k=args.smooth_k, alpha=args.smooth_alpha)
            prob_dist = prob_dist.numpy()
            
            # 调试信息：打印真实值及其概率分布前5
            topk = torch.topk(torch.tensor(prob_dist), k=5)
            # print(f"真实像素值: {true_pixel_value}，对应概率: {prob_dist[true_pixel_value]}\n 预测概率分布前5: {topk}")
            
            # 算术编码：将概率区间转化为比特流
            encoder.encode(normalize_pdf_for_arithmetic_coding(prob_dist), true_pixel_value)
            
            # 更新 xt (去噪)
            xt[0, idx] = true_pixel_id
            maskable_mask[0, idx] = False
    
    # 完成编码
    encoder.terminate()
    compressed_bits = "".join(map(str, output_bits))
    
    return compressed_bits, seq_len

def main(args):
    print(f"模型路径: {args.model_path}")
    print(f"基础模型: {args.base_model_name}")
    # 加载离散扩散模型和分词器
    tokenizer, model = load_ddm(args)
    #######################################################################
    task_prompt = "Every three values denote an RGB pixel of a flattened image. Predict the next RGB pixel based on the previous pixels."
    # 1. 准备数据
    print(f"正在从目录读取待压缩图像: {args.input_path}")
    ##TODO: 数据加载RGB顺序要改一下
    data_iterator = data_loaders.get_cifar10_iterator(
                    num_chunks=constants.NUM_CHUNKS,
                    is_channel_wised=constants.IS_CHANNEL_WISED,  # 是否分通道处理
                    is_seq=False,  # 按图像块顺序提取
                    data_path=args.input_path)
    
    # save_dir_patch = os.path.join(args.output_path, 'patches/compress')
    # os.makedirs(save_dir_patch, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, 'compressed_output.txt')
    with open(output_file, "w") as f:
        pass  # 程序启动时执行清空操作，无内容写入
    
    idx = 0  # idx是patch编号, frame_id是图像编号
    total_bits = 0
    h, w = constants.CHUNK_SHAPE_2D
    channels = 1 if constants.IS_CHANNEL_WISED else 3  # 图像块的通道数
    print(f"开始扩散压缩过程，总步数 T={args.diffusion_steps}...")
    for data, frame_id in tqdm(data_iterator):
        # 现在放到data_loaders里筛选，减少数据加载工作量
        # if frame_id in [3,6,25,0,22,12,4,13,1,11]:  # 每个类别取1张图，共10张
        # if frame_id in [1]:  # 测试用
            patch_name = f"{frame_id}_{idx % constants.PATCHES_PER_IMAGE}"
            # patch_visualize(data, save_dir_patch, patch_name)
            print(f"处理序号为 {frame_id} 的图片的第 {idx % constants.PATCHES_PER_IMAGE + 1} 个图像块...")
            seq_array = data
            seq_array = seq_array.reshape(1, h*w*channels)  # [1, h*w*c]，同时验证大小是否正确
            flattened_array = seq_array.flatten()  # 展平为一维数组
            num_str_tokens = [str(num) for num in flattened_array]  # 数字转换为字符串
            
            start_time = time.time()
            compressed_bits, seq_len = compress_image(num_str_tokens, model, tokenizer, args)
            end_time = time.time()
            
            num_bits = len(compressed_bits)
            
            ###添加哨兵比特 (Sentinel Bit)
            # 仅在末尾添加一个 '1'，作为数据结束的标志，保护有效数据的末尾 '0' 不被误删
            bits_to_write = compressed_bits + '1'
            
            total_bits += num_bits + 1
            print("\n" + "="*30)
            print(f"原始Token数: {seq_len}")
            print(f"压缩后比特数: {num_bits+1}")
            print(f"压缩耗时: {end_time - start_time:.2f}s")
            print("="*30)
            
            with open(output_file, "a") as f:
                f.write(bits_to_write + '\n')
            idx += 1
    print(f"\n所有图像块总压缩比特数: {total_bits}")
    print(f"压缩文件已保存至: {output_file}")
    
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='diffugpt-s', choices=['diffugpt-s', 'diffugpt-m', 'diffullama'])
    parser.add_argument("--base_model_name", type=str, default='gpt2', choices=['gpt2', 'gpt2-medium', 'llama'])
    parser.add_argument("--model_path", type=str, default='../Model', help="DiffuGPT path")
    parser.add_argument("--ddm_sft", type=bool, default=True, help="是否使用微调后的DiffuGPT模型")
    parser.add_argument("--checkpoint_name", type=str, default='train_ckp-4000_251225')
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--confidence_st", type=str, default='entropy', choices=['entropy', 'topk', 'simple'], help="置信度计算策略")
    parser.add_argument("--smooth_k", type=int, default=0, help="概率平滑半径")
    parser.add_argument("--smooth_alpha", type=float, default=0, help="概率平滑强度")
    parser.add_argument('--verbose', type=bool, default=False, help='打印详细过程')
    parser.add_argument("--keep_bos", type=bool, default=True, help="是否保留BOS不压缩(作为已知条件)")
    parser.add_argument("--input_path", type=str, default="../Dataset/CIFAR10")
    parser.add_argument("--output_path", type=str, default="./image_io")
    args = parser.parse_args()
    
    args.model_path = os.path.join(args.model_path, args.model_name)
    if args.ddm_sft:
        args.model_path = os.path.join(args.model_path, "ddm-sft", args.checkpoint_name)
    args.output_path = os.path.join(args.output_path, f'{args.confidence_st}_confidence', f'smooth_k{args.smooth_k}_alpha{args.smooth_alpha}')
    args.output_path = os.path.join(args.output_path, 'channel_indep') if constants.IS_CHANNEL_WISED else os.path.join(args.output_path, 'channel_corre')
    if args.ddm_sft:
        args.output_path = os.path.join(args.output_path, f'patch{constants.CHUNK_SHAPE_2D}', f'{args.model_name}_ddm-sft', f'{args.checkpoint_name}', f'diffu_step{args.diffusion_steps}')
    else:
        args.output_path = os.path.join(args.output_path, f'patch{constants.CHUNK_SHAPE_2D}', f'{args.model_name}', f'diffu_step{args.diffusion_steps}')
    return args

if __name__ == "__main__":
    set_seed(42)
    main(get_args())