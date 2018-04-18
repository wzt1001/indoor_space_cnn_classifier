from PIL import Image
import os, sys

path = "../web/final/images"
all_images = []
for path, subdirs, files in os.walk(path):
    for name in files:
        all_images.append(os.path.join(path, name))

def resize():
	for item in all_images:
		if os.path.isfile(item):
			im = Image.open(item)
			imResize = im.resize((200,150), Image.ANTIALIAS)
			imResize.save(item.replace('.png', '.jpg'), 'JPEG', quality=90)
			# os.remove(item)

resize()