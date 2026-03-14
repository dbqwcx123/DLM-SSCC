#!/bin/bash

# set -e 在后台任务中比较复杂，建议保留以便捕捉 Step 1 的错误
set -e

wait
echo "11"
python decompress_image_diffugpt.py --diffusion_steps=10 --channel=AWGN --channel_code=POLAR_K32_N64 --dataset_type=DIV2K_LR_X4

wait
echo "12"
python decompress_image_diffugpt.py --diffusion_steps=10 --channel=Rayleigh --snr_max=9 --channel_code=POLAR_K32_N64 --dataset_type=DIV2K_LR_X4

wait
echo "13"
python decompress_image_diffugpt.py --diffusion_steps=10 --channel=AWGN --channel_code=POLAR_K43_N64 --dataset_type=DIV2K_LR_X4

wait
echo "14"
python decompress_image_diffugpt.py --diffusion_steps=10 --channel=Rayleigh --snr_max=11 --channel_code=POLAR_K43_N64 --dataset_type=DIV2K_LR_X4

wait
echo "15"
python decompress_image_diffugpt.py --diffusion_steps=10 --channel=AWGN --channel_code=POLAR_K48_N64 --dataset_type=DIV2K_LR_X4

wait
echo "16"
python decompress_image_diffugpt.py --diffusion_steps=10 --channel=Rayleigh --snr_max=11 --channel_code=POLAR_K48_N64 --dataset_type=DIV2K_LR_X4

wait
echo "17"
python decompress_image_diffugpt.py --diffusion_steps=100 --channel=AWGN --snr_max=3 --channel_code=POLAR_K32_N64 --dataset_type=DIV2K_LR_X4

wait
echo "18"
python decompress_image_diffugpt.py --diffusion_steps=100 --channel=Rayleigh --snr_max=9 --channel_code=POLAR_K32_N64 --dataset_type=DIV2K_LR_X4


echo ""
echo "🎉 All steps completed successfully!"