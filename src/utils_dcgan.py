"""
Some codes from https://github.com/Newmu/dcgan_code
"""
import math
import imageio
import numpy as np
from skimage.transform import resize
from constants import NUM_TILES

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def imread(path):
    return imageio.imread(path).astype(np.float)

def get_image(image_path, image_size):
    return transform(imread(image_path), image_size)

def save_images_one_every_batch(images, grid_size, batch_size, image_path, channels=3):
    assert channels == 3 # only 3 supported at moment
    h, w = int(images.shape[1]), int(images.shape[2])
    img = np.zeros((h * int(grid_size[0]), w * int(grid_size[1]), channels))
    num_imgs = int(grid_size[0]) * int(grid_size[1])

    imgs = 0
    for idx in range(0, NUM_TILES):
        imgs += 1
        col = int(idx % grid_size[1])
        row = int(idx // grid_size[1])
        img[row*h:row*h+h, col*w:col*w+w, :] = images[idx * batch_size + 1,:,:,:]
        if imgs == num_imgs:
            print('num_imgs %d reached -> grid full. #images: %d' % (num_imgs, images.shape[0]))
            break

    return imageio.imwrite(image_path, img)


def save_images_multi(images, imagesR, subimg, grid_size, batch_size, image_path, invert=True, channels=3, maxImg=None):
    if invert:
        images = inverse_transform(images)
        imagesR = inverse_transform(imagesR)
        if subimg is not None:
            subimg = inverse_transform(subimg)

    return imsave_multi(images,imagesR,subimg, grid_size, batch_size, image_path, channels, maxImg)

def imsave_multi(images, imagesR, subimg, grid_size, batch_size, path, channels, angle=None, maxImg=None):
    assert images.shape == imagesR.shape
    h, w = int(images.shape[1]), int(images.shape[2])
    img = np.zeros((h * int(grid_size[0]), w * int(grid_size[1]), channels))

    for i in range(grid_size[0]):
        if i >= images.shape[0]:
            break
        if maxImg and i >= maxImg:
            break
        j = 0
        img[i*h:(i+1)*h, j*w:(j+1)*w, :] = images[i,:,:,:]
        j = 1
        img[i*h:(i+1)*h, j*w:(j+1)*w, :] = imagesR[i,:,:,:]
        if subimg is not None:
            for j in range(2,grid_size[1]):
                img[i*h:(i+1)*h, j*w:(j+1)*w, :] = subimg[i+batch_size*(j-2),:,:,:]

    if channels == 1:
        img = img.reshape(img.shape[0:2])
    
    return imageio.imwrite(path, img)

def save_images(images, grid_size, image_path, invert=True, channels=3,angle=None, maxImg=None):
    if invert:
        images = inverse_transform(images)
    return imsave(images, grid_size, image_path, channels,angle, maxImg)

def imsave(images, grid_size, path, channels, angle=None, maxImg=None):
    h, w = int(images.shape[1]), int(images.shape[2])
    img = np.zeros((h * int(grid_size[0]), w * int(grid_size[1]), channels))
    num_imgs = int(grid_size[0]) * int(grid_size[1])

    imgs = 0
    for idx, image in enumerate(images):
        if maxImg and idx >= maxImg:
            break
        imgs += 1
        i = int(idx % grid_size[1])
        j = int(idx // grid_size[1])
        
        if channels == 1:
            img[j*h:j*h+h, i*w:i*w+w, 0] = image 
        else:
            img[j*h:j*h+h, i*w:i*w+w, :] = image
        if imgs == num_imgs:
            print('num_imgs %d reached -> grid full. #images: %d' % (num_imgs, images.shape[0]))
            break
            
    if channels == 1:
        img = img.reshape(img.shape[0:2])

    return imageio.imwrite(path, img)

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return resize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def transform(image, npx=64):
    # npx : # of pixels width/height of image
    cropped_image = center_crop(image, npx)
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.
