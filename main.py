import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from skimage import io
from skimage import color
import numpy as np
from PIL import Image, ImageOps
from scipy import misc
import glob
import imageio
import cv2
import os
from eigenfaces import *
from cascade_detection import*
from lbp import*  

# fetching images from ./resources and converting them into grayscale

def fetch(folder_path):
    list_np_arrays = []
    for image_path in glob.glob(folder_path + "/*.png"):
        img_open = cv2.imread(image_path)
        gray = cv2.cvtColor(img_open, cv2.COLOR_BGR2GRAY)
        list_np_arrays.append(gray)

    return list_np_arrays

def fetch_and_convert(folder_path):

    list_np_arrays = []
    for image_path in glob.glob(folder_path + "/*.png"):
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


def scale_img(folder_path = "./resources", max_width = 100, max_height = 100, inter = cv2.INTER_AREA):
    
    images = fetch_and_convert(folder_path)
    maxsize = get_max_size(images)
    scaled_images_list = []
    scaling_factor = 0
    for img in images:
        '''
        (height,width) = img.shape[:2]
        ratio = min(max_width/width, max_height/height)
        #ratio = max_width / float(width)
        dim = (int(height*ratio), int(width*ratio))
        '''
        resized_img = cv2.resize(img, (max_width, max_height), interpolation = inter)
        scaled_images_list.append(resized_img)
    scaled_images_array = np.asarray(scaled_images_list)   
    return scaled_images_array

def display(arr = scale_img()):
    for img in arr:
        print(img.shape)
        images = Image.fromarray(img)
        plt.imshow(images, interpolation='nearest')
        plt.show()
        

def change_file_name(name='james_corden'):     
        if os.path.isdir("./resources/gray_scale/"):
            for i, filename in enumerate(os.listdir("./resources/gray_scale/")):
                os.rename("./resources/gray_scale/" + "/" + filename, "./resources/gray_scale/" + "/"+str(name)+'_' + str(i) + ".png")

# convert matrix to grayscale image between [0, 255]
def matrix_to_img(mat):
    img = np.copy(mat)
    img -= np.min(img) 
    img /= np.max(img) 
    img *= 255
    img = img.astype(np.uint8)
    return img

'''
train_images = scale_img()
avg_face, eigenfaces = compute_eigenfaces(train_images, (100, 100), 10)

for eigenface in eigenfaces:
    eigenface = matrix_to_img(eigenface)
    image = Image.fromarray(eigenface)
    plt.imshow(image)
    plt.show()

test_images = scale_img("./resources/test_images")
for test_image in test_images:
    image = reconstruct_img(test_image, avg_face, eigenfaces, (100, 100))
    image = Image.fromarray(image)
    plt.imshow(image)
    plt.show()
'''
#display()
#cascade_detection(fetch('./resources/training_images'))
#change_file_name()

    

def get_images_labels():
    test_labels = []
    training_labels = []
    path_train = './resources/training_images/'
    path_test = './resources/test_images'
    files_train = [os.path.splitext(filename)[0] for filename in os.listdir(path_train)]
    files_test = [os.path.splitext(filename)[0] for filename in os.listdir(path_test)]
    for trains in files_train:
        test_labels.append(trains)
    for tests in files_test:
        training_labels.append(tests)

    return test_labels, scale_img('./resources/test_images'), training_labels, scale_img('./resources/training_images')

def calculate_lbp():
    arr = []
    images = scale_img('./resources/training_images')
    counter = 0
    for img in images:
        if(counter <= 20):
            image = img.dot(0.5).astype(np.uint8)
            lbp_image = standard_lbp(image)
            arr.append(lbp_image)
        else:
            break

#get_images_labels()
#generate_lbp_histograms(arr)
#print(lbp_image)
