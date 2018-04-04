# download the raw image data from the Google Cloud
# Bolei Zhou

import os
import time
link_download = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/%s.npy'
with open('quickdraw-dataset/categories.txt') as f:
    lines = f.readlines()

os.mkdir('data')
for line in lines:
    line = line.rstrip()
    download_name = line.replace(' ','%20')
    save_name = line.replace(' ','_')
    download_link = link_download % download_name
    cmd_line = 'wget %s -O data/%s.npy' % (download_link, save_name)
    print cmd_line
    os.system(cmd_line)
    time.sleep(2)

