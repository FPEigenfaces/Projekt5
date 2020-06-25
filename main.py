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


def fetch(folder_path, max_width=100, max_height=100):
    list_np_arrays = []
    for image_path in glob.glob(folder_path + "/*.png"):
        img_open = cv2.imread(image_path)
        gray = cv2.cvtColor(img_open, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(
            gray, (max_width, max_height), interpolation=cv2.INTER_AREA)
        list_np_arrays.append(resized_img)

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


def scale_img(folder_path="./resources", max_width=100, max_height=100, inter=cv2.INTER_AREA):

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
        resized_img = cv2.resize(
            img, (max_width, max_height), interpolation=inter)
        scaled_images_list.append(resized_img)
    scaled_images_array = np.asarray(scaled_images_list)
    return scaled_images_array


def display(arr=scale_img()):
    for img in arr:
        print(img.shape)
        images = Image.fromarray(img)
        plt.imshow(images, interpolation='nearest')
        plt.show()


def change_file_name(name='james_corden', path='resources/'):
    if os.path.isdir("./resources/gray_scale/"):
        for i, filename in enumerate(os.listdir("./resources/gray_scale/")):
            os.rename("./resources/gray_scale/" + "/" + filename,
                      "./resources/gray_scale/" + "/"+str(name)+'_' + str(i) + ".png")

# convert matrix to grayscale image between [0, 255]


def matrix_to_img(mat):
    img = np.copy(mat)
    img -= np.min(img)
    img /= np.max(img)
    img *= 255
    img = img.astype(np.uint8)
    return img

# display()
# cascade_detection(fetch('./resources/training_images'))
# change_file_name()


# hier bekommt man labels und bilder der trainings und test einheiten
def get_images_labels():
    test_labels = []
    training_labels = []
    path_train = './resources/training_images/'
    path_test = './resources/test_images/'
    files_train = [os.path.splitext(filename)[0]
                   for filename in os.listdir(path_train)]
    files_test = [os.path.splitext(filename)[0]
                  for filename in os.listdir(path_test)]
    for trains in files_train:
        training_labels.append(trains)
    for tests in files_test:
        test_labels.append(tests)

    test_images = fetch(folder_path='./resources/test_images')
    training_images = fetch(folder_path='./resources/training_images')
    return test_labels, test_images, training_labels, training_images


# get_images_labels()
test_labels, test_images, train_labels, train_images = get_images_labels()


def lbp_generate_histograms_face_no_face():

    train_lbp_images = []
    test_lbp_images = []

    print('Berechne Trainings LBP Bilder...')
    for img in train_images:
        train_lbp_images.append(standard_lbp(img))

    print('Berechne Test LBP Bilder...')
    for img in test_images:
        test_lbp_images.append(standard_lbp(img))

    for test_lbp_image in test_lbp_images:
        plt.imshow(test_lbp_image, cmap='gray', vmin=0, vmax=255)
        plt.title('LBP-Image')
        plt.show()

    trains_histograms = generate_lbp_histograms(train_lbp_images)
    test_histograms = generate_lbp_histograms(test_lbp_images)
    distances = cdist(test_histograms, trains_histograms, 'cityblock')
    thrshold = 3000
    for i in range(len(test_histograms)):
        min_idx = np.argmin(distances[i])
        min_dist = np.min(distances[i])
        if min_dist <= thrshold:
            print('<%s;%s;%s;face' % (i, test_labels[i], min_dist))
        else:
            print('<%s; %s;%s;no face' % (i, test_labels[i], min_dist))

    for i in range(len(distances)):
        min_idx = np.argmin(distances[i])
        prediction = train_labels[min_idx]
        if test_labels[i][:3] not in prediction:
            print('FALSE %s;%s;%s' % (i, test_labels[i], prediction))
        else:
            print('TRUE %s;%s;%s' % (i, test_labels[i], prediction))


def eigenfaces_face_no_face():
    avg_face, eigenfaces = compute_eigenfaces(train_images, (100, 100), 15)
    '''
    for eigenface in eigenfaces:
        eigenface = matrix_to_img(eigenface.reshape(100, 100))
        image = Image.fromarray(eigenface)
        plt.imshow(image)
        plt.show()
    '''
    for i, img in enumerate(test_images):
        reconstructed_face = reconstruct_img(
            img, avg_face, eigenfaces, (100, 100))
        is_face, dist = classify_face(
            img, reconstructed_face, threshold=55000000)
        print(test_labels[i], "isface:", is_face, " | ", dist)


def predict_face_with_eigenfaces():
    counter_false = 0
    counter_true = 0
    counter = 0
    avg_face, eigenfaces = compute_eigenfaces(train_images, (100, 100), 15)
    train_proj = get_train_projection(train_images, avg_face, eigenfaces)

    for i, img in enumerate(test_images):
        counter += 1
        id = get_most_similar_face_id(img, avg_face, eigenfaces, train_proj)
        if test_labels[i][:3] not in train_labels[id]:
            print('False: %s, Predicted: %s ' %
                  (test_labels[i], train_labels[id]))
            counter_false += 1
        else:
            print('True: %s, Predicted: %s ' %
                  (test_labels[i], train_labels[id]))
            counter_true += 1

    percentage_true = (counter_true/counter)*100
    percentage_false = (counter_false/counter)*100
    print('percentage of True: %s' % (round(percentage_true)))
    print('percentage of False: %s' % (round(percentage_false)))


def crop_img(file_path):
    images = fetch(file_path)
    crop = []
    print(len(images))
    for img in images:
        crop_img = img[58:90, 35:85]
        crop.append(crop_img)
        #cv2.imshow("cropped", crop_img)
        # cv2.waitKey(2000)

    return crop


# get_images_labels()
# generate_lbp_histograms(arr)
# print(lbp_image)
lbp_generate_histograms_face_no_face()
# eigenfaces_face_no_face()
# crop_img('./resources/beard')


def beard_no_beard():
    images_train = crop_img('./resources/beard')
    images_test = crop_img('./resources/beard_test')
    train_lbp_images = []
    test_lbp_images = []

    print('Berechne Trainings LBP Bilder...')
    for img in images_train:
        train_lbp_images.append(standard_lbp(img))

    print('Berechne Test LBP Bilder...')
    for img in images_test:
        test_lbp_images.append(standard_lbp(img))

    trains_histograms = generate_lbp_histograms(train_lbp_images)
    test_histograms = generate_lbp_histograms(test_lbp_images)
    distances = cdist(test_histograms, trains_histograms, 'cityblock')
    thrshold = 590

    for i in range(len(test_histograms)):
        min_idx = np.argmin(distances[i])
        min_dist = np.min(distances[i])
        if min_dist <= thrshold:
            print('ID:%s Label:%s Distance:%s Prediction: Beard' %
                  (i, test_labels[i], min_dist))
        else:
            print('ID:%s Label:%s Distance:%s Prediction: NO Beard' %
                  (i, test_labels[i], min_dist))


# predict_face_with_eigenfaces()
# beard_no_beard() #erkennt 3 tage bart
