import argparse
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, GPT2Tokenizer

# --- 1. 解决导入路径问题 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# --- 2. 导入项目模块 ---
from train_image_diffugpt import Div2kPatchDataset as Div2kDataset
import constants
from diffu_model import load_ddm

def verify_dataset(dataset):
    print("\n" + "="*30)
    print("开始验证数据集...")
    print(f"数据集类型: {type(dataset)}")

    # --- 3. 获取样本 (适配 IterableDataset) ---
    try:
        data_iter = iter(dataset)
        sample = next(data_iter)
        print("成功通过迭代器获取到第一个样本。")
    except StopIteration:
        print("错误：数据集是空的 (StopIteration)。请检查数据路径是否正确。")
        return
    except Exception as e:
        print(f"获取样本时发生未知错误: {e}")
        return

    input_ids = sample['input_ids'].tolist()
    print(f"样本 input_ids 长度: {len(input_ids)}")
    print(f"前10个 Token ID: {input_ids[:10]}")

    # --- 4. 解码 Token ---
    # 获取 BOS Token ID (用于去除)
    bos_id = dataset.tokenizer.bos_token_id
    if bos_id is None:
        bos_id = dataset.tokenizer.eos_token_id
    
    token_ids_to_viz = input_ids
    if len(input_ids) > 0 and input_ids[0] == bos_id:
        token_ids_to_viz = input_ids[1:]
    
    print("正在解码 Token 回像素值...")
    try:
        # 将 Token ID 转回 字符串 (e.g. "128", "255")
        decoded_tokens = dataset.tokenizer.convert_ids_to_tokens(token_ids_to_viz)
        
        pixel_vals = []
        for t in decoded_tokens:
            # 清洗 Token (GPT2 Tokenizer 可能会有 'Ġ' 前缀)
            s = t.replace('Ġ', '') if isinstance(t, str) else str(t)
            try:
                pixel_vals.append(int(s))
            except ValueError:
                # 忽略无法转为数字的特殊 token
                continue
                
    except Exception as e:
        print(f"解码失败: {e}")
        return

    # --- 5. 形状检查与 Reshape ---
    h, w = constants.CHUNK_SHAPE_2D
    # 检查是否是 Channel-wise (单通道) 还是 正常 RGB (3通道)
    # 这里通过判断像素数量来自动推断
    num_pixels = len(pixel_vals)
    print(f"解码得到的有效像素数量: {num_pixels}")

    expected_rgb = h * w * 3
    expected_gray = h * w * 1

    img_array = None
    
    if num_pixels == expected_rgb:
        print(f"像素数量匹配 RGB 模式 ({h}x{w}x3)")
        # 注意 reshape 顺序必须与 Dataset 中的 flatten 顺序一致
        # 假设是 (R,G,B) 顺序或 (Row, Col, Channel)
        img_array = np.array(pixel_vals, dtype=np.uint8).reshape(h, w, 3)
    elif num_pixels == expected_gray:
        print(f"像素数量匹配单通道模式 ({h}x{w}x1)")
        img_array = np.array(pixel_vals, dtype=np.uint8).reshape(h, w)
    else:
        print(f"错误：像素数量 {num_pixels} 不符合预期。")
        print(f"预期 RGB: {expected_rgb}, 预期单通道: {expected_gray}")
        return

    # --- 6. 可视化 ---
    plt.figure(figsize=(5, 5))
    if img_array.ndim == 3:
        plt.imshow(img_array)
    else:
        plt.imshow(img_array, cmap='gray')
        
    plt.title(f"Reconstructed Patch {h}x{w}")
    plt.axis('off')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    # 默认路径根据你的描述进行了调整
    parser.add_argument("--data_path", type=str, default="../Dataset/DIV2K/DIV2K_train_LR/X4_test", help="数据文件夹路径")
    parser.add_argument("--model_path", type=str, default="../Model/diffugpt-s")
    parser.add_argument("--base_model_name", type=str, default="gpt2")
    args = parser.parse_args()

    # 1. 检查数据路径
    abs_data_path = os.path.join(project_root, args.data_path) if args.data_path.startswith("..") else args.data_path
    if not os.path.exists(abs_data_path) and not os.path.exists(args.data_path):
         print(f"警告: 数据路径 {args.data_path} 似乎不存在。尝试使用绝对路径: {abs_data_path}")
    
    # 2. 加载 Tokenizer
    tokenizer, model = load_ddm(args)

    # 3. 实例化 Dataset
    # 注意：这里参数对应上一轮提供的 Div2kIterableDataset
    print("正在实例化 Dataset...")
    dataset = Div2kDataset(
        data_path=args.data_path, # 传入路径
        tokenizer=tokenizer,
        num_chunks=5, # 测试模式只取少量数据
        is_channel_wised=constants.IS_CHANNEL_WISED
    )

    # 4. 执行验证
    verify_dataset(dataset)

if __name__ == "__main__":
    main()