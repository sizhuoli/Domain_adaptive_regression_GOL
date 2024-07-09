#  load all tifs in the folder and make a png of each tif using rgb bands
import glob
import os
import rasterio
from matplotlib import pyplot as plt
from tqdm import tqdm

folder = 'anonymous_root/france_regression/france/rgb/'
dst_folder = 'anonymous_root/france_regression/france/png_for_vis/'
all_tifs = glob.glob(folder + '*.jp2')
print(len(all_tifs))
for tif in tqdm(all_tifs):
    with rasterio.open(tif) as src:
        profile = src.profile
        img = src.read()
        img = img.transpose(1, 2, 0)
        img = img[:, :, :3]
        img = img.astype('uint8')
        img_path = os.path.join(dst_folder, os.path.basename(tif).replace('.jp2', '.png'))
        # print(img_path)
        plt.imsave(img_path, img)
