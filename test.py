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
snr_values = list(range(-3, 13, 3))

dataset_type = 'DIV2K_LR_X4'
image_indices = range(0, 10)  # 10张图像

conf_st = 'entropy'  # 'entropy' or 'simple'
channel_wise = 'corre'  # 'indep' or 'corre'

model = 'diffugpt-s'  # 'diffugpt-s' or 'diffugpt-m'
ddm_sft = True  # True or False
if ddm_sft:
    model += '_ddm-sft'
    checkpoint_dir = 'train_20251226_231454'
    mode = f"{model}/{checkpoint_dir}"
else:
    mode = model
    
diffu_step = 50
channel_decode_alg = 'ECCT'  # 'bitflip' or 'ECCT'
code_name = 'LDPC_K24_N49'

# 存储结果的字典
results = {channel: {snr: {'psnr': [], 'ssim': []} for snr in snr_values} for channel in channels}

# 计算每个图像对的PSNR和SSIM
for channel in tqdm(channels, desc='Processing channels'):
    for snr in tqdm(snr_values, desc=f'Processing SNR for {channel}', leave=False):
    # for snr in [0]: # 仅测试 SNR=0 的情况
        for idx in tqdm(image_indices, desc=f'Processing images for SNR={snr}', leave=False):
            if dataset_type == 'CIFAR10':
                orig_path = f"../Dataset/CIFAR10/cifar10_test/test_batch_{idx}.png"
            elif dataset_type == 'DIV2K_HR':
                orig_path = f"../Dataset/DIV2K/DIV2K_HR_test/08{idx+1:02d}.png"
            elif dataset_type == 'DIV2K_LR_X2':
                orig_path = f"../Dataset/DIV2K/DIV2K_LR_test/X2/09{idx+1:02d}x2.png"
            elif dataset_type == 'DIV2K_LR_X4':
                orig_path = f"../Dataset/DIV2K/DIV2K_LR_test/X4/09{idx+1:02d}x4.png"
            recon_path = f"./image_io/{dataset_type}/{conf_st}_confidence/channel_{channel_wise}/patch{constants.CHUNK_SHAPE_2D}/{mode}/diffu_step{diffu_step}/"
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
                
                if orig_img.shape != recon_img.shape:
                    from skimage.transform import resize
                    recon_img = resize(recon_img, orig_img.shape, anti_aliasing=True)
                    recon_img = (recon_img * 255).astype(np.uint8)
                
                # 手动处理完美重建，设置PSNR上限为 100 dB
                mse = np.mean((orig_img.astype(np.float64) - recon_img.astype(np.float64)) ** 2)
                if mse < 1e-10:
                    psnr_value = 100.0  # 强制设为 100 dB
                else:
                    psnr_value = psnr(orig_img, recon_img, data_range=255)

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

csv_filename = f'./_curves/{dataset_type}_{model}_step{diffu_step}.csv'
os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
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