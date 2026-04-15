import os
import argparse
import time
import numpy as np
from PIL import Image
import imagecodecs
from tqdm import tqdm
import constants

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", type=str, default='Rayleigh', choices=[None, 'AWGN', 'Rayleigh'])
    parser.add_argument("--channel_code", type=str, default='POLAR_K32_N64')
    parser.add_argument("--dataset_type", type=str, default='Kodak', choices=['Kodak', 'DIV2K_LR_X4', 'DIV2K_HR', 'Kodak'])
    parser.add_argument("--root_dir", type=str, default="./image_io")
    args = parser.parse_args()
    
    # 路径对齐：./image_io/{dataset}/JPEG_XL
    args.root_dir = os.path.join(args.root_dir, args.dataset_type, 'JPEG_XL')
    
    if args.channel:
        args.input_dir  = os.path.join(args.root_dir, f"MM_ECCT_forward/{args.channel_code}/{args.channel}")
        args.output_dir = os.path.join(args.root_dir, f"MM_ECCT_reconstruct/{args.channel_code}/{args.channel}")
    else:
        args.input_dir  = args.root_dir
        args.output_dir = args.root_dir
    
    return args

def bits_to_bytes(bit_string):
    """将由 '0' 和 '1' 组成的字符串转回字节数据"""
    # 补齐长度使其为 8 的倍数（尽管在正常的完整传输中应该是 8 的倍数）
    remainder = len(bit_string) % 8
    if remainder != 0:
        bit_string = bit_string + '0' * (8 - remainder)
        
    byte_array = bytearray()
    for i in range(0, len(bit_string), 8):
        byte_array.append(int(bit_string[i:i+8], 2))
    return bytes(byte_array)

def main(args):
    if not os.path.exists(args.input_dir):
        print(f"❌ 错误: 路径 {args.input_dir} 不存在，请检查是否已运行信道模拟程序。")
        return
    
    # 信噪比遍历 (与 DiffuGPT 对齐)
    SNR_range_test = list(range(-3, 13))
    extra_values = [0.5, 1.5, 2.5]
    SNR_range_test.extend(extra_values)
    SNR_range_test.sort()
    
    for ii, SNR in tqdm(enumerate(SNR_range_test)):
        print(f"\n{'='*30}\n📡 开始解压 SNR={SNR} 的文件...")
        input_file = os.path.join(args.input_dir, f'demo_decode_SNR_{SNR}.txt')
        save_dir_reconstructed = os.path.join(args.output_dir, f'SNR_{SNR}')
        os.makedirs(save_dir_reconstructed, exist_ok=True)
        
        if not os.path.exists(input_file):
            print(f"⚠️ 找不到文件: {input_file}，跳过该 SNR。")
            continue
            
        print(f"正在读取压缩文件: {input_file}")
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        print(f"检测到 {len(lines)} 张图像的比特流。")
        start_time = time.time()
        
        success_count = 0
        fail_count = 0
        
        for img_idx, line in enumerate(tqdm(lines)):
            raw_bit_string = line.strip()
            if not raw_bit_string:
                continue
                
            # 移除 Padding 和停止位
            stripped_padding = raw_bit_string.rstrip('0')
            if len(stripped_padding) > 0:
                bit_string = stripped_padding[:-1]
            else:
                bit_string = ""
            
            # 将 bit 流转回 bytes
            jxl_bytes = bits_to_bytes(bit_string)
            
            # --- JPEG-XL 解码 (含错误捕获机制) ---
            try:
                # 传统算法对信道噪声极其敏感，一旦头文件出错会直接抛出异常
                recon_img_array = imagecodecs.jpegxl_decode(jxl_bytes)
                success_count += 1
            except Exception as e:
                # 信道噪声导致解压失败时，生成一张灰色占位图以维持图像数量一致（计算 PSNR/SSIM 时算作极差）
                recon_img_array = np.full(constants.IMAGE_SHAPE_TEST, 128, dtype=np.uint8)
                fail_count += 1
            
            # 保存重建图像
            img_save_path = os.path.join(save_dir_reconstructed, f'image_{img_idx}.png')
            
            if recon_img_array.ndim == 2 or recon_img_array.shape[2] == 1:
                Image.fromarray(recon_img_array.squeeze(), mode='L').save(img_save_path)
            else:
                Image.fromarray(recon_img_array, mode='RGB').save(img_save_path)
                
        end_time = time.time()
        print(f"SNR={SNR} 解压完成！成功: {success_count}, 失败崩溃: {fail_count}")
        print(f"耗时: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    main(get_args())