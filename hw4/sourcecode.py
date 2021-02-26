import numpy as np
import cv2

def binarize(img, threshold):
    for i in range(512):
        for j in range(512):
            if int(img[i][j][0]) >= threshold:
                img[i][j] = [255, 255, 255]
            else:
                img[i][j] = [0, 0, 0]
    return img


def dilation(img, kernel):
    dilation = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j][0] != 0:
                for element in kernel:
                    p,q = element
                    if(i+p >= 0)and(i+p <=(img.shape[0]-1))and(j+q >= 0)and(j+q)<=(img.shape[1]-1):
                        dilation[i+p][j+q] = [255, 255, 255]
    return dilation


def erosion(img, kernel):
    erosion = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            exist = True
            for element in kernel:
                p,q = element
                if(i+p < 0)or(i+p >= img.shape[0])or(j+q < 0)or(j+q >= img.shape[1])or(img[i+p][j+q][0]==0):
                    exist = False
                    break
            if exist:
                erosion[i][j] = [255,255,255]
    return erosion


def opening(img, kernel):
    return dilation(erosion(img, kernel), kernel)


def closing(img, kernel):
    return erosion(dilation(img, kernel), kernel)


def hitandmiss(img, J_kernel, K_kernel):
    img_o = erosion(img, J_kernel)
    img_c = erosion(255-img, K_kernel)
    ham = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img_o[i][j][0] == 255 and img_c[i][j][0] == 255:
                ham[i][j] = [255, 255, 255]
    return ham


img = cv2.imread("lena.bmp")
img = binarize(img, 128)

kernel = [[-2,-1],[-2,0],[-2,1],
          [-1,-2],[-1,-1],[-1,0],[-1,1],[-1,2],
          [0,-2],[0,-1],[0,0] ,[0,1], [0,2],
          [1,-2],[1,-1],[1,0],[1,1],[1,2],
          [2,-1],[2,0],[2,1]]

J_kernel = [[0, 0], [1, 0], [0, -1]]
K_kernel = [[0, 1], [-1, 1], [-1, 0]]


cv2.imwrite("dilation.bmp",dilation(img,kernel))
cv2.imwrite("erosion.bmp",erosion(img, kernel))
cv2.imwrite("opening.bmp", opening(img, kernel))
cv2.imwrite("closing.bmp", closing(img, kernel))
cv2.imwrite("hitandmiss.bmp", hitandmiss(img, J_kernel, K_kernel))