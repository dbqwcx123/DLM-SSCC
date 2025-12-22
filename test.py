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
channels = ['AWGN']#, 'Rayleigh']
snr_values = list(range(0, 8))
image_indices = range(0, 10)  # 10张图像
smooth_k, smooth_alpha = 1, 0.1
conf_st = 'entropy'  # 'entropy' or 'topk' or 'simple'
channel_wise = 'corre'  # 'indep' or 'corre'
mode = 'diffugpt-m'  # 'diffugpt-s' or 'diffugpt-m'
diffu_step = 130
channel_decode_alg = 'ECCT'  # 'bitflip' or 'ECCT'
code_name = 'LDPC_K24_N49'

# 存储结果的字典
results = {channel: {snr: {'psnr': [], 'ssim': []} for snr in snr_values} for channel in channels}

# 计算每个图像对的PSNR和SSIM
for channel in tqdm(channels, desc='Processing channels'):
    for snr in tqdm(snr_values, desc=f'Processing SNR for {channel}', leave=False):
        for idx in tqdm(image_indices, desc=f'Processing images for SNR={snr}', leave=False):
            # 构建路径
            orig_path = f"../Dataset/CIFAR10/cifar10_test/test_batch_{idx}.png"
            recon_path = f"./image_io/{conf_st}_confidence/smooth_k{smooth_k}_alpha{smooth_alpha}/channel_{channel_wise}/patch{constants.CHUNK_SHAPE_2D}/{mode}/diffu_step{diffu_step}/"
            recon_path += f"{channel_decode_alg}_reconstruct/{code_name}/{channel}/SNR_{snr}/image_{idx}.png"
            
            # 检查路径是否存在
            if not os.path.exists(orig_path):
                print(f"Warning: Original image not found at {orig_path}")
                continue
            if not os.path.exists(recon_path):
                print(f"Warning: Reconstructed image not found at {recon_path}")
                continue
            
            # 读取图像
            try:
                orig_img = io.imread(orig_path)
                recon_img = io.imread(recon_path)
                
                # 确保图像尺寸相同
                if orig_img.shape != recon_img.shape:
                    # 调整重建图像尺寸以匹配原图
                    from skimage.transform import resize
                    recon_img = resize(recon_img, orig_img.shape, anti_aliasing=True)
                    recon_img = (recon_img * 255).astype(np.uint8)
                
                # 计算PSNR和SSIM
                psnr_value = psnr(orig_img, recon_img, data_range=255)
                ssim_value = ssim(orig_img, recon_img, data_range=255, multichannel=True, channel_axis=-1)
                
                # 存储结果
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

# 将结果保存到文本文件
csv_filename = f'./image_io/curves/results_{mode}_{conf_st}_{channel_wise}_step{diffu_step}_k{smooth_k}alpha{smooth_alpha}_add1.csv'
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
                    averages[channel]['ssim'][i]
                ])

print("处理完成！结果已保存到 csv 文件中。")