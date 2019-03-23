import sys
sys.path.append('..')
import scipy.misc
import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
from utils_dcgan import save_images_6cols
import random
from datetime import datetime

img_size = 64
num = 5
out = np.ones((img_size*(num+1),img_size*(num+1),3))*255
# chunk_num = 8

input = 'C:\\Users\\lukaszbinden\\Dropbox\\Master Thesis\\Thesis writing\\images\\images\\data_augmentation\\ex2_190214\\tmp'
output = "result"
output = os.path.join(input, output)
output = os.path.join(output, datetime.now().strftime('%Y%m%d_%H%M%S'))
if not os.path.exists(output):
	os.makedirs(output)

onlyfiles = [join(input, f) for f in listdir(input) if isfile(join(input, f))]
shuffled_index = list(range(len(onlyfiles)))
#random.seed(4285)
random.shuffle(shuffled_index)
onlyfiles = [onlyfiles[i] for i in shuffled_index]
# print(onlyfiles)

grid = [1, num]

imgs = []
for m in range(num):
	img_path = onlyfiles[m]
	img = cv2.imread(img_path)
	if img_size != img.shape[0]:
		img = cv2.resize(img, (img_size, img_size))
		print('image resized: ', img.shape)
	imgs.append(img)

save_path = os.path.join(output, "result.jpg")
print(imgs[0].shape)
save_images_6cols(imgs[0], imgs[1], imgs[2], imgs[3], imgs[4], None, grid, None, save_path, addSpacing=4)


# k = 0
# for m in range(num):
# 	img_path = onlyfiles[m]
# 	img = cv2.imread(img_path)
# 	print('img: ', img.shape)
# 	b, g, r = cv2.split(img) # get b,g,r channels
# 	img = cv2.merge([r,g,b])
# 	j = 0
# 	i = m
# 	out[img_size*0:img_size*1,img_size*(m+1):img_size*(m+2),:] = img[img_size*i:img_size*(i+1),img_size*j:img_size*(j+1),:]
# 	out[img_size*(m+1):img_size*(m+2),img_size*0:img_size*1,:] = img[img_size*i:img_size*(i+1),img_size*j:img_size*(j+1),:]
# 	#for n in range(num):
# 	#	out[img_size*(n+1):img_size*(n+2),img_size*(m+1):img_size*(m+2),:] = img[img_size*n:img_size*(n+1),img_size*(k+2):img_size*(k+3),:]
#
#
# save_path = os.join(input, "output.jpg")
# scipy.misc.imsave(save_path, out)



# for k in range(chunk_num):
#
# 	for m in range(num):
# 		img_path = '034/2_%d.png' % m
# 		img = cv2.imread(img_path)
# 		b, g, r = cv2.split(img) # get b,g,r channels
# 		img = cv2.merge([r,g,b])
# 		j = 0
# 		i = m
# 		out[img_size*0:img_size*1,img_size*(m+1):img_size*(m+2),:] = img[img_size*i:img_size*(i+1),img_size*j:img_size*(j+1),:]
# 		out[img_size*(m+1):img_size*(m+2),img_size*0:img_size*1,:] = img[img_size*i:img_size*(i+1),img_size*j:img_size*(j+1),:]
# 		for n in range(num):
# 			out[img_size*(n+1):img_size*(n+2),img_size*(m+1):img_size*(m+2),:] = img[img_size*n:img_size*(n+1),img_size*(k+2):img_size*(k+3),:]
#
#
# 	save_path = '034'+'/%d.png'%k
# 	scipy.misc.imsave(save_path, out)
		


	# for a in xrange(1,11):
	# 	img_path = 'mnist_old/013/340102_test_sep%d%d.png'% (a,m)
	# 	if not os.path.exists(img_path): continue
	# 	img = cv2.imread(img_path)
	# 	b, g, r = cv2.split(img) # get b,g,r channels
	# 	img = cv2.merge([r,g,b])
	# 	out[img_size*a:img_size*(a+1),img_size*m:img_size*(m+1),:] = img[img_size*i:img_size*(i+1),img_size*j:img_size*(j+1),:]

	# 	img_path = 'mnist_old/013/340102_test_sep%d%d.png'% (m,a)
	# 	if not os.path.exists(img_path): continue
	# 	img = cv2.imread(img_path)
	# 	b, g, r = cv2.split(img) # get b,g,r channels
	# 	img = cv2.merge([r,g,b])
	# 	out[img_size*m:img_size*(m+1),img_size*a:img_size*(a+1),:] = img[img_size*i:img_size*(i+1),img_size*j:img_size*(j+1),:] 


# scipy.misc.imsave('figure_mnist/3_new.png', img) 


