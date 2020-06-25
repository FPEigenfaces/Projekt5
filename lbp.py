import numpy as np


def standard_lbp(input_image):

    lbp_image = np.zeros(
        (input_image.shape[0]-2, input_image.shape[1]-2)).astype(np.uint8)

    for row in range(1, input_image.shape[0]-1):
        for column in range(1, input_image.shape[1]-1):

            center = input_image[row][column]
            lbp_code = 0
            if input_image[row-1][column+1] >= center:
                lbp_code += 1*2**0
            if input_image[row][column+1] >= center:
                lbp_code += 1*2**1
            if input_image[row+1][column+1] >= center:
                lbp_code += 1*2**2
            if input_image[row+1][column] >= center:
                lbp_code += 1*2**3
            if input_image[row+1][column-1] >= center:
                lbp_code += 1*2**4
            if input_image[row][column-1] >= center:
                lbp_code += 1*2**5
            if input_image[row-1][column-1] >= center:
                lbp_code += 1*2**6
            if input_image[row-1][column] >= center:
                lbp_code += 1*2**7

            lbp_image[row-1][column-1] = lbp_code  # row-1 und col -1
    return lbp_image


def generate_lbp_histograms(lbp_image_list, num_bins=2**8):

    hists = np.zeros((len(lbp_image_list), num_bins))
    for i in range(len(lbp_image_list)):
        tmp = lbp_image_list[i].flatten()
        tmp_bincount = np.bincount(tmp)
        hists[i] = tmp_bincount
    return hists

