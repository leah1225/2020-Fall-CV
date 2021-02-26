import numpy as np
import cv2
import copy
import os, sys
from math import log, sqrt

# Previous Functions
def dilation(img, kernel):
    dilation = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j][0] > 0:
                max = 0
                for element in kernel:
                    p,q = element
                    if(i+p >= 0)and(i+p < img.shape[0])and(j+q >= 0)and(j+q < img.shape[1]):
                        if img[i+p][j+q][0] > max:
                            max = img[i+p][j+q][0]
                for element in kernel:
                    p,q = element
                    if(i+p >= 0)and(i+p < img.shape[0])and(j+q >= 0)and(j+q < img.shape[1]):
                        dilation[i+p][j+q] = max
    return dilation

def erosion(img, kernel):
    erosion = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            min = 256
            exist = True
            for element in kernel:
                p,q = element
                if(i+p >= 0)and(i+p < img.shape[0])and(j+q >= 0)and(j+q < img.shape[1]):
                    if img[i+p][j+q][0] == 0:
                        exist = False
                        break
                    if img[i+p][j+q][0] < min:
                        min = img[i+p][j+q][0]
            exist = True            
            for element in kernel:
                p,q = element
                if(i+p >= 0)and(i+p < img.shape[0])and(j+q >= 0)and(j+q < img.shape[1]):
                    if img[i+p][j+q][0] == 0:
                        exist = False
                        break
                
                if(i+p >= 0)and(i+p < img.shape[0])and(j+q >= 0)and(j+q < img.shape[1])and(exist):
                    erosion[i+p][j+q] = min
    return erosion

def opening(img, kernel):
    return dilation(erosion(img, kernel),kernel)

def closing(img, kernel):
    return erosion(dilation(img, kernel), kernel)

# noise generator
def generate_Gaussian_noise(img, mu, sigma, amplitude):
    return img + amplitude * np.random.normal(mu, sigma, img.shape)

def generate_salt_and_pepper_noise(img, low, high, threshold):
    prob_map = np.random.uniform(low, high, img.shape)
    img_sp = copy.deepcopy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if prob_map[i][j][0] < threshold:
                img_sp[i][j] = 0
            elif prob_map[i][j][0] > 1 - threshold:
                img_sp[i][j] = 255
    return img_sp

def box_filter(img, filter_size):            
    img_fil = np.zeros(shape=(img.shape[0] - filter_size + 1, img.shape[1] - filter_size + 1, 3))
    for i in range(img_fil.shape[0]):
        for j in range(img_fil.shape[1]):
            img_fil[i][j] = np.mean(img[i: i + filter_size, j: j + filter_size])
            
    if img_fil.shape != img.shape:
        b = int((img.shape[0] - img_fil.shape[0])/2)
        img_fil = cv2.copyMakeBorder(img_fil, b, b, b, b, cv2.BORDER_REFLECT)
        
    return img_fil

def median_filter(img, filter_size):
    img_fil = np.zeros(shape=(img.shape[0] - filter_size + 1, img.shape[1] - filter_size + 1, 3))
    for i in range(img_fil.shape[0]):
        for j in range(img_fil.shape[1]):
            img_fil[i][j] = np.median(img[i: i + filter_size, j: j + filter_size])
            
    if img_fil.shape != img.shape:
        b = int((img.shape[0] - img_fil.shape[0])/2)
        img_fil = cv2.copyMakeBorder(img_fil, b, b, b, b, cv2.BORDER_REFLECT)
        
    return img_fil


def normalize(img):
    return (img - img.min())*(1 - 0)/(255-0)

def getSNR(img_orig, img_noisy):
    
    orig_norml = normalize(img_orig)
    noisy_norml = normalize(img_noisy)
    
    var1 = orig_norml.var()
    var2 = (noisy_norml - orig_norml).var()
    
    return 20 * log(sqrt(var1)/sqrt(var2), 10)

img = cv2.imread('lena.bmp')

# Use octagon as kernel and set the orgin is at the center
kernel = [
    [-2, -1], [-2, 0], [-2, 1],
    [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
    [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
    [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
    [2, -1], [2, 0], [2, 1]]

# Gaussian noise with amplitude = 10
print('Gaussian noise with amplitude = 10')

img_gauss_10 = generate_Gaussian_noise(img, 0, 1, 10)
cv2.imwrite('lena.gaussian.10.bmp', img_gauss_10)
print(getSNR(img, img_gauss_10))

img_gauss_10_box_3 = box_filter(img_gauss_10, 3)
cv2.imwrite('lena.gaussian.10.box.3x3.bmp', img_gauss_10_box_3)
print(getSNR(img, img_gauss_10_box_3))

img_gauss_10_box_5 = box_filter(img_gauss_10, 5)
cv2.imwrite('lena.gaussian.10.box.5x5.bmp', img_gauss_10_box_5)
print(getSNR(img, img_gauss_10_box_5))

img_gauss_10_med_3 = median_filter(img_gauss_10, 3)
cv2.imwrite('lena.gaussian.10.median.3x3.bmp', img_gauss_10_med_3)
print(getSNR(img, img_gauss_10_med_3))

img_gauss_10_med_5 = median_filter(img_gauss_10, 5)
cv2.imwrite('lena.gaussian.10.median.5x5.bmp', img_gauss_10_med_5)
print(getSNR(img, img_gauss_10_med_5))

img_gauss_10_close_open = opening(closing(img_gauss_10, kernel), kernel)
cv2.imwrite('lena.gaussian.10.close.open.bmp', img_gauss_10_close_open)
print(getSNR(img, img_gauss_10_close_open))

img_gauss_10_open_close = closing(opening(img_gauss_10, kernel), kernel)
cv2.imwrite('lena.gaussian.10.open.close.bmp', img_gauss_10_open_close)
print(getSNR(img, img_gauss_10_open_close))



# Gaussian noise with amplitude = 30
print('Gaussian noise with amplitude = 30')

img_gauss_30 = generate_Gaussian_noise(img, 0, 1, 30)
cv2.imwrite('lena.gaussian.30.bmp', img_gauss_30)
print(getSNR(img, img_gauss_30))


img_gauss_30_box_3 = box_filter(img_gauss_30, 3)
cv2.imwrite('lena.gaussian.30.box.3x3.bmp', img_gauss_30_box_3)
print(getSNR(img, img_gauss_30_box_3))

img_gauss_30_box_5 = box_filter(img_gauss_30, 5)
cv2.imwrite('lena.gaussian.30.box.5x5.bmp', img_gauss_30_box_5)
print(getSNR(img, img_gauss_30_box_5))

img_gauss_30_med_3 = median_filter(img_gauss_30, 3)
cv2.imwrite('lena.gaussian.30.median.3x3.bmp', img_gauss_30_med_3)
print(getSNR(img, img_gauss_30_med_3))

img_gauss_30_med_5 = median_filter(img_gauss_30, 5)
cv2.imwrite('lena.gaussian.30.median.5x5.bmp', img_gauss_30_med_5)
print(getSNR(img, img_gauss_30_med_5))

img_gauss_30_close_open = opening(closing(img_gauss_30, kernel), kernel)
cv2.imwrite('lena.gaussian.30.close.open.bmp', img_gauss_30_close_open)
print(getSNR(img, img_gauss_30_close_open))

img_gauss_30_open_close = closing(opening(img_gauss_30, kernel), kernel)
cv2.imwrite('lena.gaussian.30.open.close.bmp', img_gauss_30_open_close)
print(getSNR(img, img_gauss_30_open_close))



# salt-and-pepper noise with threshold = 0.05
print('salt-and-pepper noise with threshold = 0.05')

img_sp_05 = generate_salt_and_pepper_noise(img, 0, 1, 0.05)
cv2.imwrite('lena.sp.05.bmp', img_sp_05)
print(getSNR(img, img_sp_05))

img_sp_05_box_3 = box_filter(img_sp_05, 3)
cv2.imwrite('lena.sp.05.box.3x3.bmp', img_sp_05_box_3)
print(getSNR(img, img_sp_05_box_3))

img_sp_05_box_5 = box_filter(img_sp_05, 5)
cv2.imwrite('lena.sp.05.box.5x5.bmp', img_sp_05_box_5)
print(getSNR(img, img_sp_05_box_5))

img_sp_05_med_3 = median_filter(img_sp_05, 3)
cv2.imwrite('lena.sp.05.median.3x3.bmp', img_sp_05_med_3)
print(getSNR(img, img_gauss_30_med_3))

img_sp_05_med_5 = median_filter(img_sp_05, 5)
cv2.imwrite('lena.sp.05.median.5x5.bmp', img_sp_05_med_5)
print(getSNR(img, img_sp_05_med_5))

img_sp_05_close_open = opening(closing(img_sp_05, kernel), kernel)
cv2.imwrite('lena.sp.05.close.open.bmp', img_sp_05_close_open)
print(getSNR(img, img_sp_05_close_open))

img_sp_05_open_close = closing(opening(img_sp_05, kernel), kernel)
cv2.imwrite('lena.sp.05.open.close.bmp', img_sp_05_open_close)
print(getSNR(img, img_sp_05_open_close))



# salt-and-pepper noise with threshold = 0.1
print('salt-and-pepper noise with threshold = 0.1')

img_sp_10 = generate_salt_and_pepper_noise(img, 0, 1, 0.1)
cv2.imwrite('lena.sp.10.bmp', img_sp_10)
print(getSNR(img, img_sp_10))

img_sp_10_box_3 = box_filter(img_sp_10, 3)
cv2.imwrite('lena.sp.10.box.3x3.bmp', img_sp_10_box_3)
print(getSNR(img, img_sp_10_box_3))

img_sp_10_box_5 = box_filter(img_sp_10, 5)
cv2.imwrite('lena.sp.10.box.5x5.bmp', img_sp_10_box_5)
print(getSNR(img, img_sp_10_box_5))

img_sp_10_med_3 = median_filter(img_sp_10, 3)
cv2.imwrite('lena.sp.10.median.3x3.bmp', img_sp_10_med_3)
print(getSNR(img, img_sp_10_med_3))

img_sp_10_med_5 = median_filter(img_sp_10, 5)
cv2.imwrite('lena.sp.10.median.5x5.bmp', img_sp_10_med_5)
print(getSNR(img, img_sp_10_med_5))

img_sp_10_close_open = opening(closing(img_sp_10, kernel), kernel)
cv2.imwrite('lena.sp.10.close.open.bmp', img_sp_10_close_open)
print(getSNR(img,img_sp_10_close_open))

img_sp_10_open_close = closing(opening(img_sp_10, kernel), kernel)
cv2.imwrite('lena.sp.10.open.close.bmp', img_sp_10_open_close)
print(getSNR(img, img_sp_10_open_close))