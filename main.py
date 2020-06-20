import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from skimage import io
from skimage import color
import numpy as np
from PIL import Image
from scipy import misc
import glob
import imageio
import cv2
import os
# fetching images from ./resources and converting them into grayscale


def fetch_and_convert():

    list_np_arrays = []
    for image_path in glob.glob("./resources/*.png"):
        img_open = cv2.imread(image_path)
        gray = cv2.cvtColor(img_open, cv2.COLOR_BGR2GRAY)
        list_np_arrays.append(gray)
        # gray.save('./resources/gray_scale/%s' %os.path.basename(image_path)) optional
        list_np_arrays.append(gray)

    np_arrs = np.asarray(list_np_arrays)

    return np_arrs


def get_max_size(arr):
    maxsize = (0, 0)
    for img in arr:
        images = Image.fromarray(img)
        if images.size > maxsize:
            maxsize = images.size

    return maxsize


def scale_img(max_width = 500, max_height = 500, inter = cv2.INTER_AREA):
    images = fetch_and_convert()
    maxsize = get_max_size(images)
    scaled_images_list = []
    scaling_factor = 0
    for img in images:
        (height,width) = img.shape[:2]
        ratio = min(max_width/width, max_height/height)
        #ratio = max_width / float(width)
        dim = (int(height*ratio), int(width*ratio))
        resized_img = cv2.resize(img, dim, interpolation = inter)
        scaled_images_list.append(resized_img)
    scaled_images_array = np.asarray(scaled_images_list)   
    return scaled_images_array

def display():
    arr = scale_img()
    for img in arr:
        print(img.shape)
        images = Image.fromarray(img)
        plt.imshow(images, interpolation='nearest')
        plt.show()
        

def change_file_name(name='james_corden'):     
        if os.path.isdir("./resources/gray_scale/"):
            for i, filename in enumerate(os.listdir("./resources/gray_scale/")):
                os.rename("./resources/gray_scale/" + "/" + filename, "./resources/gray_scale/" + "/"+str(name)+'_' + str(i) + ".png")

display()
#change_file_name()