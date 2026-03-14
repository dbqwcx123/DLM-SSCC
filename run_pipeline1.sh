#!/bin/bash

# set -e 在后台任务中比较复杂，建议保留以便捕捉 Step 1 的错误
set -e


# wait
# echo "constants.IMAGE_SHAPE_TEST 改为 (32, 32, 3)"
# CUDA_VISIBLE_DEVICES=1 python decompress_image_diffugpt.py --diffusion_steps=100 --channel=AWGN --dataset_type=CIFAR10

# wait
# CUDA_VISIBLE_DEVICES=1 python decompress_image_diffugpt.py --diffusion_steps=100 --channel=Rayleigh --dataset_type=CIFAR10


wait
echo "constants.IMAGE_SHAPE_TEST 改为 (256, 256, 3)"
CUDA_VISIBLE_DEVICES=1 python decompress_image_diffugpt.py --diffusion_steps=100 --channel=AWGN --dataset_type=DIV2K_LR_X4

wait
CUDA_VISIBLE_DEVICES=1 python decompress_image_diffugpt.py --diffusion_steps=100 --channel=Rayleigh --dataset_type=DIV2K_LR_X4


echo ""
echo "🎉 All steps completed successfully!"