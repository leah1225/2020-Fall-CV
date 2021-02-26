import numpy as np
import cv2

#octal kernel 3,5,5,5,3
kernel = [[-2,-1],[-2,0],[-2,1],
          [-1,-2],[-1,-1],[-1,0],[-1,1],[-1,2],
          [0,-2],[0,-1],[0,0] ,[0,1], [0,2],
          [1,-2],[1,-1],[1,0],[1,1],[1,2],
          [2,-1],[2,0],[2,1]]

img = cv2.imread("lena.bmp")

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


cv2.imwrite("dilation.bmp", dilation(img, kernel))
cv2.imwrite("erosion.bmp", erosion(img, kernel))
cv2.imwrite("opening.bmp", opening(img, kernel))
cv2.imwrite("closing.bmp", closing(img, kernel))