#!/usr/bin/python
import cv2
import tensorflow as tf
import numpy as np
import glob
from Utils.utils import make_square_image
import skimage.io
import skimage.transform
import pandas as pd
randInt = np.random.randint
"""Class file to prepare data and labels to pass to Model(DCGAN)
Yet to add mean normalization
"""
class Data_Prep:
    def __init__(self,path,batch_size,nImgs,img_size):
        self.path = path
        self.filenames = []
        self.batch_size = batch_size
        self.img_size = img_size
        self.nImgs = nImgs
        self.labels_G = None
        self.labels_D = None
        self.get_filenames()
        self.load_labels()

    def get_filenames(self):
        """Function to read filenames of all images in path to self.filenames"""
        self.filenames = pd.read_csv('../image_names.csv',header=None)[0] #Only one column, hence [0] loads all filenames in self.filenames
        self.nImgs = len(self.filenames)

    def read_batch(self,overfit):
        """ Function to read a batch of images. Normalizes images to [-1,1]
            Fixed shape of output to (64,64)
            Returns:
                curr_imgs: np array of squared images(to img_size)
                Images are normalized to the range [-1,1]
                """
        curr_imgs = []
        right_captions = []
        wrong_captions = []
        if overfit:
            idx = range(self.batch_size)
        else:
            idx = np.random.choice(self.nImgs,self.batch_size)
            idx_ = np.random.choice(self.nImgs,self.batch_size)
        for i,j in zip(idx,idx_):
            curr_img = skimage.io.imread(self.path + self.filenames[i] + '.jpg')
            curr_img = skimage.transform.resize(curr_img,(64,64))
            right_captions.append(np.load('../captions_uniskip/' + self.filenames[i] + '.npy')[randInt(10)])
            wrong_captions.append(np.load('../captions_uniskip/' + self.filenames[j] + '.npy')[randInt(10)])
            #curr_img = curr_img/255.
            curr_imgs.append(curr_img)
            #not added mean normalization
        curr_imgs = np.array(curr_imgs).astype('float32')
        right_captions = np.array(right_captions)
        wrong_captions = np.array(wrong_captions)
        return curr_imgs,right_captions,wrong_captions

    def process(self,img):
        img = make_square_image(img,64)
        img = (img/127.5) - 1.
        return img

    def load_labels(self):
        ones = np.ones((self.batch_size,1))
        zeros = np.zeros((self.batch_size,1))
        self.labels_G = ones
        self.labels_D = np.vstack([zeros,ones])
