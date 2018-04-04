import numpy as np
import skimage.io

categories = ['ladder', 'eye', 'pool', 'yoga', 'zebra', 'brain']

# TODO: output into a HTML so it is easy to see all the samples
for category in categories:
    data = np.load('data/%s.npy' % category)
    print category
    for i in range(10):
        data_instance = data[data.shape[0]-1-i]
        data_instance = data_instance.reshape((28,28))
        skimage.io.imsave('test_image/%s_%04d.jpg'%(category, i), data_instance)

