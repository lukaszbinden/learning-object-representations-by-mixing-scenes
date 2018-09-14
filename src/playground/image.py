import tensorflow as tf
from PIL import Image
import numpy as np
import os

def crop(path, input, height, width, k, area):
    im = Image.open(input)
    imgwidth, imgheight = im.size
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            try:
                #o = a.crop(area)
                o = a
                loc = os.path.join(path,"PNG", "IMG-%s.png" % k)
                print('save to ' + loc)
                o.save(loc)
            except Exception as e:
                print(e)
                pass
            k +=1


path = '..\images'
img = '..\images\lake.png'

crop(path, img, 100, 100, 1, 0)
