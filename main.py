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
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

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

    
#hier bekommt man labels und bilder der trainings und test einheiten
def get_images_labels():
    test_labels = []
    training_labels = []
    path_train = './resources/training_images/'
    path_test = './resources/test_images'
    files_train = [os.path.splitext(filename)[0] for filename in os.listdir(path_train)]
    files_test = [os.path.splitext(filename)[0] for filename in os.listdir(path_test)]
    for trains in files_train:
        training_labels.append(trains)
    for tests in files_test:
        test_labels.append(tests)

    return test_labels, scale_img('./resources/test_images'), training_labels, scale_img('./resources/training_images')


test_labels, test_images, train_labels, train_images = get_images_labels()

def lbp_generate_histograms_face_no_face():
    
    train_lbp_images = []
    test_lbp_images = []

    print('Berechne Trainings LBP Bilder')
    for img in train_images:
        train_lbp_images.append(standard_lbp(img))
    
    print('Berechne Test LBP Bilder')
    for img in test_images:
        test_lbp_images.append(standard_lbp(img))
    
    for test_lbp_image in test_lbp_images:
        plt.imshow(test_lbp_image, cmap='gray', vmin=0, vmax=255)
        plt.title('LBP-Image')
        plt.show()
    
    trains_histograms = generate_lbp_histograms(train_lbp_images)
    test_histograms = generate_lbp_histograms(test_lbp_images)
    distances = cdist(test_histograms,trains_histograms,'cityblock')
    thrshold = 1000
    print(len(distances))
    for i in range(len(distances)):
        min_idx = np.argmin(distances[i])
        min_dist = np.min(distances[i])
        if min_dist <= thrshold:
            print('<%s; %s; %s;face'%(i,test_labels[i],min_dist))
        else:
            print('<%s; %s; %s;no face'%(i,test_labels[i],min_dist))    
#get_images_labels()
#generate_lbp_histograms(arr)
#print(lbp_image)
#lbp_generate_histograms_face_no_face()