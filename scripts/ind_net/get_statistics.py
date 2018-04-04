with open('quickdraw-dataset/categories.txt') as f:
    lines = f.readlines()

print 'category no.%d' % len(lines)


import numpy as np
stat = []
for line in lines:
    line  = line.rstrip()
    category = line.replace(' ','_')
    file_np = 'data/%s.npy' % category
    data_category = np.load(file_np)
    print '%s: %d' % (category, data_category.shape[0])
    stat.append('%s %d' % (category, data_category.shape[0]))

with open('stat_all.txt','w') as f:
    f.write('\n'.join(stat))
