#!/bin/bash

# set -e 在后台任务中比较复杂，建议保留以便捕捉 Step 1 的错误
set -e

# wait
# echo "1"
# python MM_ECCT_forward.py --gpus=0 --channel=AWGN --code_k=32 --diffu_step=10

# wait
# echo "2"
# python MM_ECCT_forward.py --gpus=0 --channel=AWGN --code_k=43 --diffu_step=10

# wait
# echo "3"
# python MM_ECCT_forward.py --gpus=0 --channel=AWGN --code_k=48 --diffu_step=10

# wait
# echo "4"
# python MM_ECCT_forward.py --gpus=0 --channel=AWGN --code_k=32 --diffu_step=100

# wait
# echo "5"
# python MM_ECCT_forward.py --gpus=0 --channel=AWGN --code_k=32 --diffu_step=500

# wait
# echo "6"
# python decompress_image_diffugpt.py --gpus=0 --channel=AWGN --channel_code=POLAR_K32_N64 --diffu_steps=10 --snr_max=4

# wait
# echo "7"
# python decompress_image_diffugpt.py --gpus=0 --channel=AWGN --channel_code=POLAR_K43_N64 --diffu_steps=10 --snr_max=5

# wait
# echo "8"
# python decompress_image_diffugpt.py --gpus=0 --channel=AWGN --channel_code=POLAR_K48_N64 --diffu_steps=10 --snr_max=5

wait
echo "9"
python decompress_image_diffugpt.py --gpus=0 --channel=AWGN --channel_code=POLAR_K32_N64 --diffu_steps=100 --snr_max=4

wait
echo "10"
python decompress_image_diffugpt.py --gpus=0 --channel=AWGN --channel_code=POLAR_K32_N64 --diffu_steps=500 --snr_max=3


echo ""
echo "🎉 All steps completed successfully!"