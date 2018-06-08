##B-351 Final Project
##Captcha Recogniton w/ TensorFlow


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from PIL import Image, ImageChops, ImageEnhance
import PIL

import os
import numpy as np

import argparse
import sys
import time
import math
from captch_model import data_set





#Batch process to resize all single character captchas
def resize_images():
    cwd = os.getcwd()

    for x in "abcdefghijklmnopqrstuvwxyz":
        print("Starting:", x)
        filename = "/test4/" + x + "/test"
        for y in range(10):
            filename = filename + str(y) + ".jpg"
            resize_image(cwd, filename)
            filename = "/test4/" + x + "/test"
        
#For resizing single character captchas
def resize_image(cwd, filename):
    im = Image.open(cwd + filename)
    out = im.resize((70,70))
    out.save(cwd + filename)



#Grayscale images
def recolor_images():
    cwd = os.getcwd()
    for x in range(1000):
        filename = "/test/" + "slice_test" + str(x)+ "_"
        for y in range(1,6):
            filename1 = filename + str(y) + ".png"
            recolor_image(cwd, filename1)



def recolor_image(cwd, filename):
    im = Image.open(cwd + filename)
    gray = im.convert('L')
    gray.save(cwd + filename)

recolor_images()

#Functions to slice catpchas into equal number of pieces
def enum(iterable, start = 1):
    n = start
    for i in iterable:
        yield n, i
        n += 1

#Remove white spae from border of captcha
def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def long_slice(image_path, out_name, outdir, slice_size):
    img = Image.open(image_path)
    img = trim(img)
    img = img.resize((200,70))
    img = img.transpose(PIL.Image.ROTATE_90)
    width, height = img.size
    upper = 0
    slices = int(math.ceil(height/slice_size))#How many pieces will there be?

    for i, slice in enum(range(slices)):
        left = 0
        upper = upper
        if i == slices:
            lower = height
        else:
            lower = int(i * slice_size)        
        bbox = (left, upper, width, lower)
        working_slice = img.crop(bbox)
        working_slice = working_slice.transpose(PIL.Image.ROTATE_270)
        working_slice = working_slice.resize((70,70))
        upper += slice_size
  
        working_slice.save(os.path.join(outdir, "slice_" + out_name + "_" + str(i)+".png"))





#Work through the directory and slice captchas
def slice_images():
    cwd = os.getcwd()
    for x in range(1000):
        long_slice(cwd + "/test/test" + str(x) + ".jpg", "test" + str(x), cwd + "/test", 40)
        






    
#REad k,v filestore
def read_captcha_kv_store(path):
    keystore = open(path, 'r').read()
    records = keystore.split("\n")
    result = []
    for record in records:
        result.append(record.split(','))
    return result





