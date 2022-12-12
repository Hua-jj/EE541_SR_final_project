import os
import numpy as np
import torch
import cv2
import h5py

# NOTE: all datasets are saved in the folder /dataset

def find_sizes(src):
    # function to find the minimum sizes of images in the input src folder
    # return (width, height)
    min_height, min_width = float('inf'), float('inf')
    for f in os.listdir(src):
        img = cv2.imread(src+f,cv2.IMREAD_UNCHANGED)
        if img.shape[0] < min_height:
            min_height = img.shape[0]
        elif img.shape[1] < min_width:
            min_width = img.shape[1]
    return min_width, min_height


def matching_pairs(lr_src,hr_src):
    # function to match the LR images and HR images
    print('matching pairs')
    lr = os.listdir(lr_src)
    hr = os.listdir(hr_src)
    lr.sort()
    hr.sort()
    for i in range(len(lr)):
        if lr[i][:-6] != hr[i][:-4]:
            print('img not matched! ' + lr[i])
    return lr, hr


def bgr2ycbcr_y(img, y=True):
    # img: input array in uint8 (BGR, not RGB!)
    # reference: https://sistenix.com/rgb2ycbcr.html
    # the returned data types is float64

    Y = (np.dot(img,[25.064,129.057,65.738]))/256+16

    if y:
        return Y
    else:
        Cb = (np.dot(img,[112.439,-74.494,-37.945]))/256+128
        Cr = (np.dot(img,[-18.285,-94.154,112.439]))/256+128

        new_img = np.zeros(img.shape)
        new_img[:,:,0] = Y
        new_img[:,:,1] = Cb
        new_img[:,:,2] = Cr
        return new_img


def convert_rgb_to_y(lr_src,hr_src,lr,hr,lr_size,hr_size):
    # convert all LR and HR images from RGB to YCbCr and extract the y-channel
    print('converting')
    imgs_lr, imgs_hr = [],[]

    # for lr images
    for f in lr:
        # return BGR in uint8
        img = cv2.imread(lr_src+f,cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, lr_size)
        img_y = bgr2ycbcr_y(img)/255
        imgs_lr += [torch.from_numpy(img_y)]
  
    # for hr images
    for f in hr:
        img = cv2.imread(hr_src+f,cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, hr_size)
        img_y = bgr2ycbcr_y(img)/255
        imgs_hr += [torch.from_numpy(img_y)]
  
    # returned y-channels float64
    return imgs_lr, imgs_hr


def save_h5py(lr, hr, name, lr_size, hr_size,length=800,train=False,name_val=None):
    print('saving')
    # lr_size, hr_size: (width, height)
    # should be stored in (height, width)?
    db = h5py.File(name,'w')

    # lr_size = (width, height)
    if train:
        db_lr = db.create_dataset('lr',(length-100,lr_size[1],lr_size[0]),dtype='float32')
        db_hr = db.create_dataset('hr',(length-100,hr_size[1],hr_size[0]),dtype='float32')

        for i in range(len(lr[:-100])):
            db_lr[i] = lr[i]
            db_hr[i] = hr[i]
        db.close()
        print('file is saved at {}'.format(name))
    
        db_v = h5py.File(name_val,'w')
        db_lr_v = db_v.create_dataset('lr',(100,lr_size[1],lr_size[0]),dtype='float32')
        db_hr_v = db_v.create_dataset('hr',(100,hr_size[1],hr_size[0]),dtype='float32')

        a = np.arange(0,100)
        b = np.arange(700,800)
        for x,y in zip(a,b):
            db_lr_v[x] = lr[y]
            db_hr_v[x] = hr[y]
        db_v.close()
        print('file is saved at {}'.format(name_val))

    else:
        db_lr = db.create_dataset('lr',(length,lr_size[1],lr_size[0]),dtype='float32')
        db_hr = db.create_dataset('hr',(length,hr_size[1],hr_size[0]),dtype='float32')
        for i in range(len(lr)):
            db_lr[i] = lr[i]
            db_hr[i] = hr[i]
        db.close()
        print('file is saved at {}'.format(name))


def preprocess(lr_src,hr_src,h5file,x=2,h5file_val=None,lr_sizes=None,hr_sizes=None):
    returned = False
    if lr_sizes is None or hr_sizes is None:
        # compute sizes for images
        print('width, height')
        lr_sizes = find_sizes(lr_src)
        print(lr_sizes)

        # hr sizes
        hr_sizes = (int(lr_sizes[0]*x), int(lr_sizes[1]*x))
        print(hr_sizes)
        returned = True

    # convert images
    lr,hr = matching_pairs(lr_src,hr_src)
    lr_train,hr_train = convert_rgb_to_y(lr_src,hr_src,lr,hr,lr_sizes,hr_sizes)

    # save in h5 format
    # (lr, hr, name, lr_size, hr_size,length=800,train=False,name_val=None)
    if h5file_val is not None:
        save_h5py(lr_train,hr_train,h5file, lr_sizes, hr_sizes, len(lr_train),train=True,name_val=h5file_val)
    else:
        save_h5py(lr_train,hr_train,h5file, lr_sizes, hr_sizes, len(lr_train))
  

    if returned:
        return (lr_sizes, hr_sizes)


# example usage:
# prepare data for x2 type
lr_s2, hr_s2 = preprocess('dataset/DIV2K_train_LR_bicubic/X2/','dataset/DIV2K_train_HR/',
                          'dataset/train_x2.h5py',x=2,h5file_val='dataset/valid_x2.h5py')
preprocess('dataset/DIV2K_valid_LR_bicubic/X2/','dataset/DIV2K_valid_HR/',
           'dataset/test_x2.h5py', x=2,lr_sizes=lr_s2,hr_sizes=hr_s2)