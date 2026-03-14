import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import io
import csv
import constants
from tqdm import tqdm

# 配置参数
channels = ['AWGN', 'Rayleigh']
snr_values = list(range(-3, 13, 2))

dataset_type = 'CIFAR10'  # CIFAR10, DIV2K_LR_X4, DIV2K_HR, Kodak

model = 'igpt-s'  # diffugpt-s, igpt-s, JPEG_XL
ddm_sft = False  # True or False
if ddm_sft:
    model += '_ddm-sft'
    checkpoint_dir = 'train_20251226_231454'
    mode = f"{model}/{checkpoint_dir}"
else:
    mode = model
    
diffu_step = 10
channel_decode_alg = 'MM_ECCT'  # SCL, BP, ECCT, MM_ECCT
code_name = 'POLAR_K32_N64'

if 'diffugpt' in model:
    csv_filename = f'./_curves/{dataset_type}/{channel_decode_alg}/{code_name}/{model}_step{diffu_step}.csv'
else:
    csv_filename = f'./_curves/{dataset_type}/{channel_decode_alg}/{code_name}/{model}.csv'
os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

# 存储结果的字典
results = {channel: {snr: {'psnr': [], 'ssim': []} for snr in snr_values} for channel in channels}

# 计算每个图像对的PSNR和SSIM
for channel in tqdm(channels, desc='Processing channels'):
    for snr in tqdm(snr_values, desc=f'Processing SNR for {channel}', leave=False):
    # for snr in [0]: # 仅测试 SNR=0 的情况
        for idx in range(10):
            orig_path = f"./image_io/{dataset_type}/gt/image_{idx}.png"
            if 'diffugpt' in model:
                recon_path = f"./image_io/{dataset_type}/patch{constants.CHUNK_SHAPE_2D}/{mode}/diffu_step{diffu_step}/"
            elif 'igpt' in model:
                recon_path = f"./image_io/{dataset_type}/patch{constants.CHUNK_SHAPE_2D}/{mode}/"
            elif 'JPEG' in model:
                recon_path = f"./image_io/{dataset_type}/{mode}/"
            else:
                print("model 类型有误！")
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
                
                orig_img_int = orig_img.astype(np.uint8)
                recon_img_int = recon_img.astype(np.uint8)
                
                # 计算绝对差值矩阵
                diff = np.abs(orig_img_int - recon_img_int)
                # 手动处理完美重建，设置PSNR上限为 100 dB
                mse = np.mean(diff ** 2)
                if mse < 1e-10:
                    psnr_value = 100.0
                else:
                    psnr_value = psnr(orig_img, recon_img, data_range=255)

                # ================= 新增：寻找并输出错误像素 =================
                # if mse > 0:
                #     # np.where 找出所有差值大于 0 的索引
                #     error_rows, error_cols, error_channels = np.where(diff > 0)
                #     total_errors = len(error_rows)
                #     print(f"  [!] 发现 {total_errors} 个通道值有差异。")
                    
                #     for i in range(total_errors):
                #         r = error_rows[i]
                #         c = error_cols[i]
                #         ch = error_channels[i]
                        
                #         orig_val = orig_img[r, c, ch]
                #         recon_val = recon_img[r, c, ch]
                #         error_val = diff[r, c, ch]
                        
                #         print(f"      -> 第 {r:3d} 行, 第 {c:3d} 列, 第 {ch} 通道 | "
                #               f"原图: {orig_val:3d}, 重建: {recon_val:3d} | 差值: {error_val}")
                    
                #     if total_errors > 10:
                #         print(f"      ... (省略其余 {total_errors - 10} 个差异点)")
                # else:
                #     print("  [✓] 像素完全一致，零误差重建！")
                # ============================================================
                
                ssim_value = ssim(orig_img, recon_img, data_range=255, multichannel=True, channel_axis=-1)
                
                results[channel][snr]['psnr'].append(psnr_value)
                results[channel][snr]['ssim'].append(ssim_value)
                
            except Exception as e:
                print(f"Error processing image pair (channel={channel}, SNR={snr}, idx={idx}): {e}")

# 计算平均值
averages = {channel: {'psnr': [], 'ssim': []} for channel in channels}
for channel in channels:
    for snr in snr_values:
        psnr_list = results[channel][snr]['psnr']
        ssim_list = results[channel][snr]['ssim']
        if psnr_list and ssim_list:  # 确保列表不为空
            avg_psnr = np.mean(psnr_list)
            avg_ssim = np.mean(ssim_list)

            averages[channel]['psnr'].append(avg_psnr)
            averages[channel]['ssim'].append(avg_ssim)
        else:
            print(f"Warning: No data for {channel}, SNR={snr}")

with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Channel', 'SNR', 'Avg PSNR', 'Avg SSIM'])
    
    for channel in channels:
        for i, snr in enumerate(snr_values):
            if averages[channel]['psnr']:  # 确保列表不为空
                writer.writerow([
                    channel, 
                    snr, 
                    averages[channel]['psnr'][i],
                    averages[channel]['ssim'][i],
                ])

print("处理完成！结果已保存到 csv 文件中。")