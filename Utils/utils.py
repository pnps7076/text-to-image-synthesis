#!/usr/bin/python
import cv2
import numpy as np
import scipy.misc
from PIL import Image

#file to include utility functions for data preparation
def resize(image, width=None, height=None, inter=cv2.INTER_LANCZOS4):
    '''resize the image such that its aspect ration is preserved. Only height or width is required as the
    other parameter is calculated using the aspect raito of the given image.
    input parameters
      image_name:- str. Image name along with the full path
      width:- integer. The width of the output image
      height:- integer. If widht is provided then height is not needed. The height of the output image.
    output
      returns the path + resized image name (str) which is resized to the image widht or height given in the input.
    '''

    # read the image
    #image = cv2.imread(image_name)

    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image_name

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image_name
    #ext = image_name.split(".")[-1]
    #if ext == "png" or ext == "PNG":
    #    resized_img_name = image_name.split(".")[0] + "_resized.png"
    #else:
    #    resized_img_name = image_name.split(".")[0] + "_resized.jpg"
    #cv2.imwrite(resized_img_name, resized)
    return resized


# making square image
def make_square_image(img, size=None):
    '''
    Resize the image to a desired height or width. Once this is done pad the shorter side such that both the height and the width are equal.
    Only height or width is required as the other parameter will be the same as we are constructing square images.
    input parameters
      image_name:- str. Image name along with the full path
      width:- integer. The width of the output image
      height:- integer. If widht is provided then height is not needed. The height of the output image.
    output
      returns the path + squared image name:- str.
    '''

    if size == None:
        return "Please return a valid size"
    else:
        #ext = image_name.split(".")[-1]
        #img = cv2.imread(image_name)
        h, w = img.shape[:2]
        if h >= w:
            resized_img = resize(img, height=size)
            #resized_img = cv2.imread(resized_img_name)
            r_h, r_w = resized_img.shape[:2]
            padding_size = size - r_w
            if img.ndim==3:
                square_image = np.pad(resized_img, ((
                0, 0), (padding_size / 2, padding_size / 2 + padding_size % 2), (0, 0)), "constant")
            else:
                square_image = np.pad(resized_img, ((
                0, 0), (padding_size / 2, padding_size / 2 + padding_size % 2)), "constant")
        else:
            resized_img = resize(img, width=size)
            #resized_img = cv2.imread(resized_img_name)
            r_h, r_w = resized_img.shape[:2]
            padding_size = size - r_h
            if img.ndim==3:
                square_image = np.pad(resized_img, ((
                padding_size / 2, padding_size / 2 + padding_size % 2), (0, 0), (0, 0)), "constant")
            else:
                square_image = np.pad(resized_img, ((
                padding_size / 2, padding_size / 2 + padding_size % 2), (0, 0)), "constant")

        #if ext == "png" or ext == "PNG":
        #    square_img_name = image_name.split(".")[0] + "_square_img." + ext
        #else:
        #    square_img_name = image_name.split(".")[0] + "_square_img.jpg"

        #cv2.imwrite(square_img_name, square_image)
        return square_image
