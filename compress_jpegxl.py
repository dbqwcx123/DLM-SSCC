import os
import argparse
import time
import numpy as np
from PIL import Image
import imagecodecs
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, default="DIV2K_LR_X4", choices=['CIFAR10', 'DIV2K_LR_X4', 'DIV2K_HR', 'Kodak'])
    parser.add_argument("--input_path", type=str, default="../Dataset")
    parser.add_argument("--output_path", type=str, default="./image_io")
    args = parser.parse_args()
    
    # 与你的 DiffuGPT 完全一致的路径解析逻辑
    if args.dataset_type == "CIFAR10":
        args.input_path = os.path.join(args.input_path, "CIFAR10", "cifar10_test")
    elif args.dataset_type == "DIV2K_HR":
        args.input_path = os.path.join(args.input_path, "DIV2K", "DIV2K_HR_test")
    elif args.dataset_type == "DIV2K_LR_X2":
        args.input_path = os.path.join(args.input_path, "DIV2K", "DIV2K_LR_test/X2")
    elif args.dataset_type == "DIV2K_LR_X4":
        args.input_path = os.path.join(args.input_path, "DIV2K", "DIV2K_LR_test/X4")
    elif args.dataset_type == "DIV2K_LR_test":
        args.input_path = os.path.join(args.input_path, "DIV2K", "DIV2K_LR_test/test")
    elif args.dataset_type == "Kodak":
        args.input_path = os.path.join(args.input_path, "Kodak", "test_unified")
    else:
        args.input_path = os.path.join(args.input_path, args.dataset_type)
    
    # 输出路径: ./image_io/CIFAR10/JPEG_XL
    args.output_path = os.path.join(args.output_path, args.dataset_type, 'JPEG_XL')
    return args

def main(args):
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, 'compress_output.txt')
    
    # 清空已存在文件
    with open(output_file, "w") as f: pass

    # 获取图像列表
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = sorted([f for f in os.listdir(args.input_path) if f.lower().endswith(valid_exts)])
    
    if not image_files:
        print(f"❌ 未在 {args.input_path} 找到支持的图像文件。")
        return

    print(f"🚀 开始 JPEG-XL 编码压缩...")
    print(f"图像数据集路径: {args.input_path}")
    print(f"输出文件: {output_file}")
    
    total_bits = 0
    total_pixels = 0
    start_time = time.time()
    
    with open(output_file, "a") as f:
        for img_name in tqdm(image_files):
            img_path = os.path.join(args.input_path, img_name)
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)
            
            # 记录像素总量以计算 bpp
            h, w, c = img_array.shape
            total_pixels += h * w * c
            
            # 使用 imagecodecs 进行 JPEG-XL 编码
            # JPEG-XL 特有参数：distance。0 表示无损，大于0表示有损（数值越大压缩率越高，质量越低，1.0 左右为视觉无损）
            jxl_bytes = imagecodecs.jpegxl_encode(img_array, distance=0, lossless=True, effort=9)  # effort为努力程度，取值1-9
            
            # 将 bytes 转为 bit 字符串 '0101...'
            bits = ''.join(f'{b:08b}' for b in jxl_bytes)
            bits_to_write = bits + '1'
            f.write(bits_to_write + '\n')
            
            total_bits += len(bits_to_write)
            
    end_time = time.time()
    
    print("\n" + "="*40)
    print(f"总处理图像数: {len(image_files)}")
    print(f"总压缩比特数: {total_bits}")
    print(f"压缩比特率 (bpsp): {(total_bits / total_pixels):.3f}")
    print(f"总耗时: {end_time - start_time:.2f}s")
    print("="*40 + "\n")

if __name__ == "__main__":
    main(get_args())