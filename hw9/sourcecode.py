import warnings
warnings.filterwarnings("ignore")
import numpy as np
import cv2
from math import sqrt

# Roberts operator
def Roberts(img, threshold):
    img_rob = np.zeros((img.shape[0], img.shape[1]))
    Gx = np.zeros((img.shape[0], img.shape[1]))
    Gy = np.zeros((img.shape[0], img.shape[1]))
    
    for i in range(img.shape[0]-1):
        for j in range(img.shape[1]-1):
            Gx[i][j] = (-1)*img[i][j][0] + 0*img[i][j+1][0] + 0*img[i+1][j][0] + 1*img[i+1][j+1][0]
            Gy[i][j] = 0*img[i][j][0] + (-1)*img[i][j+1][0] + 1*img[i+1][j][0] + 0*img[i+1][j+1][0]
            
    for i in range(img_rob.shape[0]):
        for j in range(img_rob.shape[1]):
            if sqrt(Gx[i][j]**2 + Gy[i][j]**2) < threshold:
                img_rob[i][j] = 255
    return img_rob

# Prewitt edge detector
def Prewitt(img, threshold):
    img_pre = np.zeros((img.shape[0], img.shape[1]))
    Gx = np.zeros((img.shape[0], img.shape[1]))
    Gy = np.zeros((img.shape[0], img.shape[1]))
    
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            Gx[i][j] = (-1)*(img[i-1][j-1][0]+img[i-1][j][0]+img[i-1][j+1][0])+1*(img[i+1][j-1][0]+img[i+1][j][0]+img[i+1][j+1][0])
            Gy[i][j] = (-1)*(img[i-1][j-1][0]+img[i][j-1][0]+img[i+1][j-1][0])+1*(img[i-1][j+1][0]+img[i][j+1][0]+img[i+1][j+1][0])
            
    for i in range(img_pre.shape[0]):
        for j in range(img_pre.shape[1]):
            if sqrt(Gx[i][j]**2 + Gy[i][j]**2) < threshold:
                img_pre[i][j] = 255
    return img_pre

# Sobel edge detector
def Sobel(img, threshold):
    img_sob = np.zeros((img.shape[0], img.shape[1]))
    Gx = np.zeros((img.shape[0], img.shape[1]))
    Gy = np.zeros((img.shape[0], img.shape[1]))
    
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            Gx[i][j] = (-1)*(img[i-1][j-1][0]+2*img[i-1][j][0]+img[i-1][j+1][0])+1*(img[i+1][j-1][0]+2*img[i+1][j][0]+img[i+1][j+1][0])
            Gy[i][j] = (-1)*(img[i-1][j-1][0]+2*img[i][j-1][0]+img[i+1][j-1][0])+1*(img[i-1][j+1][0]+2*img[i][j+1][0]+img[i+1][j+1][0])
            
    for i in range(img_sob.shape[0]):
        for j in range(img_sob.shape[1]):
            if sqrt(Gx[i][j]**2 + Gy[i][j]**2) < threshold:
                img_sob[i][j] = 255
    return img_sob

# Frei and Chen gradient operator
def FreiandChen(img, threshold):
    img_fac = np.zeros((img.shape[0], img.shape[1]))
    Gx = np.zeros((img.shape[0], img.shape[1]))
    Gy = np.zeros((img.shape[0], img.shape[1]))
    
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            Gx[i][j] = (-1)*(img[i-1][j-1][0]+sqrt(2)*img[i-1][j][0]+img[i-1][j+1][0])+1*(img[i+1][j-1][0]+sqrt(2)*img[i+1][j][0]+img[i+1][j+1][0])
            Gy[i][j] = (-1)*(img[i-1][j-1][0]+sqrt(2)*img[i][j-1][0]+img[i+1][j-1][0])+1*(img[i-1][j+1][0]+sqrt(2)*img[i][j+1][0]+img[i+1][j+1][0])
            
    for i in range(img_fac.shape[0]):
        for j in range(img_fac.shape[1]):
            if sqrt(Gx[i][j]**2 + Gy[i][j]**2) < threshold:
                img_fac[i][j] = 255
    return img_fac

# Kirsch compass operator
def Kirsch(img, threshold):
    img_kir = np.zeros((img.shape[0], img.shape[1]))
    mask = [[-3, -3, 5, -3, 0, 5, -3, -3, 5],
            [-3, 5, 5, -3, 0, 5, -3, -3, -3],
            [5, 5, 5, -3, 0, -3, -3, -3, -3],
            [5, 5, -3, 5, 0, -3, -3, -3, -3],
            [5, -3, -3, 5, 0, -3, 5, -3, -3],
            [-3, -3, -3, 5, 0, -3, 5, 5, -3],
            [-3, -3, -3, -3, 0, -3, 5, 5, 5],
            [-3, -3, -3, -3, 0, 5, -3, 5, 5]]
    
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            neighbor = [img[i-1][j-1][0], img[i-1][j][0], img[i-1][j+1][0],
                        img[i][j-1][0], img[i][j][0], img[i][j+1][0],
                        img[i+1][j-1][0], img[i+1][j][0], img[i+1][j+1][0]]
            kmaps = np.zeros(8)
            
            for k in range(8):
                kmaps[k] = sum([mask[k][tmp] * neighbor[tmp] for tmp in range(9)])
                
            img_kir[i][j] = max(kmaps)
            
    img_kir = np.where(img_kir >= threshold, 0, 255)

    return img_kir

# Robinson compass operator
def Robinson(img, threshold):
    img_robi = np.zeros((img.shape[0], img.shape[1]))
    mask = [[-1, 0, 1, -2, 0, 2, -1, 0, 1],
            [0, 1, 2, -1, 0, 1, -2, -1, 0],
            [1, 2, 1, 0, 0, 0, -1, -2, -1],
            [2, 1, 0, 1, 0, -1, 0, -1, -2],
            [1, 0, -1, 2, 0, -2, 1, 0, -1],
            [0, -1, -2, 1, 0, -1, 2, 1, 0],
            [-1, -2, -1, 0, 0, 0, 1, 2, 1],
            [-2, -1, 0, -1, 0, 1, 0, 1, 2]]
    
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            neighbor = [img[i-1][j-1][0], img[i-1][j][0], img[i-1][j+1][0],
                        img[i][j-1][0], img[i][j][0], img[i][j+1][0],
                        img[i+1][j-1][0], img[i+1][j][0], img[i+1][j+1][0]]
            kmaps = np.zeros(8)
            
            for k in range(8):
                kmaps[k] = sum([mask[k][tmp] * neighbor[tmp] for tmp in range(9)])
                
            img_robi[i][j] = max(kmaps)
    
    img_robi = np.where(img_robi >= threshold, 0, 255)
    
    return img_robi

# Nevatia and Babu 5×5 operator
def NevatiaandBabu(img, threshold):
    img_nab = np.zeros((img.shape[0], img.shape[1]))
    mask = [[100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 0, 0, 0, 0, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
            [100, 100, 100, 100, 100, 100, 100, 100, 78, -32, 100, 92, 0, -92, -100, 32, -78, -100, -100, -100, -100, -100, -100, -100, -100],
            [100, 100, 100, 32, -100, 100, 100, 92, -78, -100, 100, 100, 0, -100, -100, 100, 78, -92, -100, -100, 100, -32, -100, -100, -100],
            [-100, -100, 0, 100, 100, -100, -100, 0, 100, 100, -100, -100, 0, 100, 100, -100, -100, 0, 100, 100, -100, -100, 0, 100, 100],
            [-100, 32, 100, 100, 100, -100, -78, 92, 100, 100, -100, -100, 0, 100, 100, -100, -100, -92, 78, 100, -100, -100, -100, -32, 100],
            [100, 100, 100, 100, 100, -32, 78, 100, 100, 100, -100, -92, 0, 92, 100, -100, -100, -100, -78, 32, -100, -100, -100, -100, -100]]
    
    for i in range(2, img.shape[0]-2):
        for j in range(2, img.shape[1]-2):
            neighbor = [img[i-2][j-2][0], img[i-2][j-1][0], img[i-2][j][0], img[i-2][j+1][0], img[i-2][j+2][0],
                        img[i-1][j-2][0], img[i-1][j-1][0], img[i-1][j][0], img[i-1][j+1][0], img[i-1][j+2][0],
                        img[i][j-2][0], img[i][j-1][0], img[i][j][0], img[i][j+1][0], img[i][j+2][0],
                        img[i+1][j-2][0], img[i+1][j-1][0], img[i+1][j][0], img[i+1][j+1][0], img[i+1][j+2][0],
                        img[i+2][j-2][0], img[i+2][j-1][0], img[i+2][j][0], img[i+2][j+1][0], img[i+2][j+2][0]]
            kmaps = np.zeros(6)
            
            for k in range(6):
                kmaps[k] = sum([mask[k][tmp] * neighbor[tmp] for tmp in range(25)])
            
            img_nab[i][j] = max(kmaps)
                
    img_nab = np.where(img_nab >= threshold, 0, 255)
    
    return img_nab

img = cv2.imread('lena.bmp')

cv2.imwrite('img_rob30.bmp',Roberts(img, 30))

cv2.imwrite('img_pre24.bmp',Prewitt(img, 24))

cv2.imwrite('img_sob38.bmp',Sobel(img, 38))

cv2.imwrite('img_fac30.bmp',FreiandChen(img, 30))

cv2.imwrite('img_kir135.bmp',Kirsch(img, 135))

cv2.imwrite('img_robi43.bmp',Robinson(img, 43))

cv2.imwrite('img_nab12500.bmp',NevatiaandBabu(img, 12500))