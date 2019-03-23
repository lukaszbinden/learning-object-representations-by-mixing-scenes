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


def save_images_6cols(imgs1, imgs2, imgs3, imgs4, imgs5, imgs6, grid_size, batch_size, image_path, invert=True, channels=3, maxImg=None, addSpacing=None):
    if invert:
        imgs1 = inverse_transform(imgs1)
        imgs2 = inverse_transform(imgs2)
        if imgs3 is not None:
            imgs3 = inverse_transform(imgs3)
        if imgs4 is not None:
            imgs4 = inverse_transform(imgs4)
        if imgs5 is not None:
            imgs5 = inverse_transform(imgs5)
        if imgs6 is not None:
            imgs6 = inverse_transform(imgs6)

    return imsave_6cols(imgs1, imgs2, imgs3, imgs4, imgs5, imgs6, grid_size, batch_size, image_path, channels, maxImg, addSpacing)


def imsave_6cols(imgs1, imgs2, imgs3, imgs4, imgs5, imgs6, grid_size, batch_size, path, channels, maxImg=None, addSpacing=None):
    #assert imgs1.shape == imgs2.shape and imgs2.shape == imgs3.shape and imgs3.shape == imgs4.shape
    #assert imgs4.shape == imgs5.shape
    #assert imgs5.shape == imgs6.shape
    h, w = int(imgs1.shape[0]), int(imgs1.shape[1])

    spacing = addSpacing if addSpacing else 0
    img = np.ones((h * int(grid_size[0]), w * int(grid_size[1]) + 4 + (spacing * 3), channels)) # (spacing * 4) -> assuming 6 images are given

    for i in range(grid_size[0]):
        if i >= imgs1.shape[0]:
            break
        if maxImg and i >= maxImg:
            break
        j = 0
        img[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = imgs1
        j = 1
        img[i * h:(i + 1) * h, j * w + 4:(j + 1) * w + 4, :] = imgs2
        if imgs3 is not None:
            j = 2
            img[i * h:(i + 1) * h, j * w + spacing*2:(j + 1) * w + spacing*2, :] = imgs3
        if imgs4 is not None:
            j = 3
            img[i * h:(i + 1) * h, j * w + spacing*3:(j + 1) * w + spacing*3, :] = imgs4
        if imgs5 is not None:
            j = 4
            img[i * h:(i + 1) * h, j * w + spacing*4:(j + 1) * w + spacing*4, :] = imgs5
        if imgs6 is not None:
            j = 5
            img[i * h:(i + 1) * h, j * w + spacing*5:(j + 1) * w + spacing*5, :] = imgs6

    if channels == 1:
        assert 1==2, "not supported at moment"
        # img = img.reshape(img.shape[0:2])

    return imageio.imwrite(path, img)

def save_images_5cols(imgs1, imgs2, imgs3, imgs4, imgs5, grid_size, batch_size, image_path, invert=True, channels=3, maxImg=None):
    if invert:
        imgs1 = inverse_transform(imgs1)
        imgs2 = inverse_transform(imgs2)
        imgs3 = inverse_transform(imgs3)
        imgs4 = inverse_transform(imgs4)
        imgs5 = inverse_transform(imgs5)

    return imsave_5cols(imgs1, imgs2, imgs3, imgs4, imgs5, grid_size, batch_size, image_path, channels, maxImg)


def imsave_5cols(imgs1, imgs2, imgs3, imgs4, imgs5, grid_size, batch_size, path, channels, maxImg=None):
    assert imgs1.shape == imgs2.shape and imgs2.shape == imgs3.shape and imgs3.shape == imgs4.shape
    assert imgs4.shape == imgs5.shape
    h, w = int(imgs1.shape[1]), int(imgs1.shape[2])
    img = np.zeros((h * int(grid_size[0]), w * int(grid_size[1]), channels))

    for i in range(grid_size[0]):
        if i >= imgs1.shape[0]:
            break
        if maxImg and i >= maxImg:
            break
        j = 0
        img[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = imgs1[i, :, :, :]
        j = 1
        img[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = imgs2[i, :, :, :]
        j = 2
        img[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = imgs3[i, :, :, :]
        j = 3
        img[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = imgs4[i, :, :, :]
        j = 4
        img[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = imgs5[i, :, :, :]

    if channels == 1:
        assert 1==2, "not supported at moment"
        # img = img.reshape(img.shape[0:2])

    return imageio.imwrite(path, img)


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
