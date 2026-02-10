import os
import glob
from PIL import Image
from tqdm import tqdm
import multiprocessing

# ================= 配置 =================
SOURCE_PATH = '../Dataset/DIV2K/DIV2K_HR_unified/train'  # 原大图路径
SAVE_PATH = '../Dataset/DIV2K/DIV2K_HR_p256/train'    # 保存切片的路径
PATCH_SIZE = 256   # 建议切成 64x64 或 128x128，不要切 16x16 (太碎了文件系统受不了)
STRIDE = 256       # 步长，等于 PATCH_SIZE 表示不重叠
# =======================================

def process_one_image(img_path):
    if not os.path.exists(SAVE_PATH):
        try:
            os.makedirs(SAVE_PATH, exist_ok=True)
        except:
            pass
            
    img_name = os.path.basename(img_path).split('.')[0]
    try:
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        
        # 开始切图
        cnt = 0
        for y in range(0, h - PATCH_SIZE + 1, STRIDE):
            for x in range(0, w - PATCH_SIZE + 1, STRIDE):
                box = (x, y, x + PATCH_SIZE, y + PATCH_SIZE)
                patch = img.crop(box)
                
                save_name = f"{img_name}_p{cnt}.png"
                patch.save(os.path.join(SAVE_PATH, save_name))
                cnt += 1
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

if __name__ == '__main__':
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        
    img_list = sorted(glob.glob(os.path.join(SOURCE_PATH, '*.png')))
    print(f"Found {len(img_list)} images. Start patching...")
    
    # 使用多进程加速切图
    pool = multiprocessing.Pool(processes=8)
    list(tqdm(pool.imap(process_one_image, img_list), total=len(img_list)))
    pool.close()
    pool.join()
    
    print("Done! Patches saved to:", SAVE_PATH)