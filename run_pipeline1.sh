#!/bin/bash

# set -e 在后台任务中比较复杂，建议保留以便捕捉 Step 1 的错误
set -e

# wait
# echo "1"
# python MM_ECCT_forward.py --gpus=0 --channel=AWGN --mode=JPEG_XL

# wait
# echo "3"
# python MM_ECCT_forward.py --gpus=0 --channel=Rayleigh --mode=JPEG_XL


# wait
# echo "5"
# python decompress_jpegxl.py --channel=AWGN

# wait
# echo "7"
# python decompress_jpegxl.py --channel=Rayleigh


# wait
# echo "9"
# python compress_image_diffugpt.py --gpus=1


# wait
# echo "11"
# python MM_ECCT_forward.py --gpus=1 --channel=AWGN

# wait
# echo "13"
# python MM_ECCT_forward.py --gpus=1 --channel=Rayleigh


wait
echo "15"
python decompress_image_diffugpt.py --gpus=1 --channel=AWGN --snr_max=4

wait
echo "17"
python decompress_image_diffugpt.py --gpus=1 --channel=Rayleigh --snr_max=8


echo ""
echo "🎉 All steps completed successfully!"