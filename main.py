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
# fetching images from ./resources and converting them into grayscale


def fecth_and_convert():
    list_np_arrays = []
    for image_path in glob.glob("./resources/*.png"):
        img_open = cv2.imread(image_path)
        gray = cv2.cvtColor(img_open, cv2.COLOR_BGR2GRAY)
        list_np_arrays.append(gray)
        # gray.save('./resources/gray_scale/%s' %os.path.basename(image_path)) optional
        list_np_arrays.append(gray)

    np_arrs = np.asarray(list_np_arrays)

    return np_arrs


def display(arr):
    for img in arr:
        images = Image.fromarray(img)
        plt.imshow(images, interpolation='nearest')
        plt.show()
        break


arrs = fecth_and_convert()
display(arrs)
