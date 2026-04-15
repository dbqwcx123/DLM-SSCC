import os
import numpy as np
import csv
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
import constants
from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 配置参数
channels = ['AWGN']#, 'Rayleigh']
SNR_range_test = list(range(0, 3))

dataset_type = 'CIFAR10'  # CIFAR10, DIV2K_LR_X4, Kodak

model = 'diffugpt-s'  # diffugpt-s, igpt-s, JPEG_XL, DLPR
ddm_sft = True  # True or False
if ddm_sft:
    model += '_ddm-sft'
    checkpoint_dir = 'train_20251226_231454'
    mode = f"{model}/{checkpoint_dir}"
else:
    mode = model
    
diffu_step = 100
channel_decode_alg = 'MM_ECCT'  # SCL, BP, ECCT, MM_ECCT
code_name = 'POLAR_K32_N64'

if 'diffugpt' in model:
    csv_filename = f'./_curves/{dataset_type}/{channel_decode_alg}/{code_name}/{model}_step{diffu_step}.csv'
else:
    csv_filename = f'./_curves/{dataset_type}/{channel_decode_alg}/{code_name}/{model}.csv'
os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

# ======================================================================
SDOP_PSNR_THRESHOLD = 30

# 存储结果的字典 (改为存储各指标的具体数值)
results = {
    channel: {
        snr: {'mse': [], 'ssim': []} 
        for snr in SNR_range_test
    } 
    for channel in channels
}

# 计算指标
for channel in tqdm(channels, desc='Processing channels'):
    for snr in tqdm(SNR_range_test, desc=f'Processing SNR for {channel}', leave=False):
    # for snr in [0]: # 仅测试 SNR=0 的情况
        for idx in range(10):
            orig_path = f"./image_io/{dataset_type}/gt/image_{idx}.png"
            if 'diffugpt' in model:
                recon_path = f"./image_io/{dataset_type}/patch{constants.CHUNK_SHAPE_2D}/{mode}/diffu_step{diffu_step}/"
            elif 'igpt' in model:
                recon_path = f"./image_io/{dataset_type}/patch{constants.CHUNK_SHAPE_2D}/{mode}/"
            else:
                recon_path = f"./image_io/{dataset_type}/{mode}/"
            recon_path += f"{channel_decode_alg}_reconstruct/{code_name}/{channel}/SNR_{snr}/image_{idx}.png"
            
            if not os.path.exists(orig_path):
                print(f"Warning: Original image not found at {orig_path}")
                continue
            if not os.path.exists(recon_path):
                print(f"Warning: Reconstructed image not found at {recon_path}")
                continue
            
            try:
                orig_img = io.imread(orig_path)
                recon_img = io.imread(recon_path)
                
                # ----------------------------------------------------
                # 1. 记录 MSE 用于后续计算 Corrected PSNR
                # ----------------------------------------------------
                diff = np.abs(orig_img.astype(np.float32) - recon_img.astype(np.float32))
                mse = np.mean(diff ** 2)
                results[channel][snr]['mse'].append(mse)
                
                # ----------------------------------------------------
                # 2. 计算标准 SSIM
                # ----------------------------------------------------
                ssim_val = ssim(orig_img, recon_img, data_range=255, multichannel=True, channel_axis=-1)
                results[channel][snr]['ssim'].append(ssim_val)
                               
            except Exception as e:
                print(f"Error processing image pair (channel={channel}, SNR={snr}, idx={idx}): {e}")

# ================= 汇总与写入 CSV =================
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # 更新了 CSV 表头
    writer.writerow(['Channel', 'SNR', 'PSNR', 'SSIM', 'SDOP'])
    
    for channel in channels:
        for snr in SNR_range_test:
            mse_list = results[channel][snr]['mse']
            ssim_list = results[channel][snr]['ssim']
            
            if not mse_list:
                continue
                
            # --- 1. 计算 PSNR ---
            avg_mse = np.mean(mse_list)
            if avg_mse < 1e-10:
                corrected_psnr = 100.0
            else:
                corrected_psnr = 10 * np.log10((255.0 ** 2) / avg_mse)
                
            # --- 2. 计算 SSIM ---
            avg_ssim = np.mean(ssim_list)
            
            # --- 3. 计算 SDOP (语义失真中断概率) ---
            # 统计 MSE 大于容忍阈值的图片数量
            SDOP_MSE_THRESHOLD = (255.0 ** 2) / (10 ** (SDOP_PSNR_THRESHOLD / 10))
            outage_count = sum(1 for val in mse_list if val > SDOP_MSE_THRESHOLD)
            sdop = outage_count / len(mse_list)
            
            writer.writerow([
                channel, snr, 
                f"{corrected_psnr:.4f}", 
                f"{avg_ssim:.4f}", 
                f"{sdop:.4f}"
            ])

print("处理完成！结果已保存到 csv 文件中。")