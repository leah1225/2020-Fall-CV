import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import copy

# Laplace Mask 1
def Lapalce1(img, threshold):
    img_lap = np.zeros((img.shape[0], img.shape[1]))
    lap_map = np.zeros((img.shape[0], img.shape[1]))
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i == 0 and j == 0: # 左上
                mask = [-4, 1, 1, 0]
                neighbor = [img[i][j][0], img[i][j+1][0],
                            img[i+1][j][0], img[i+1][j+1][0]]
            elif i == 0 and j == img.shape[1]-1: # 右上
                mask = [1, -4, 0, 1]
                neighbor = [img[i][j-1][0], img[i][j][0],
                            img[i+1][j-1][0], img[i+1][j][0]]
            elif i == 0 and j != 0 and j != img.shape[1]-1: #第一排、非左上飛右上
                mask = [1, -4, 1, 0, 1, 0]
                neighbor = [img[i][j-1][0], img[i][j][0], img[i][j+1][0],
                            img[i+1][j-1][0], img[i+1][j][0], img[i+1][j+1][0]]
            elif i == img.shape[0]-1 and j == 0: # 左下
                mask = [1, 0, -4, 1]
                neighbor = [img[i-1][j][0], img[i-1][j+1][0],
                            img[i][j][0], img[i][j+1][0]]
            elif i == img.shape[0]-1 and j == img.shape[1]-1: # 右下
                mask = [0, 1, 1, -4]
                neighbor = [img[i-1][j-1][0], img[i-1][j][0],
                            img[i][j-1][0], img[i][j][0]]
            elif i == img.shape[0]-1 and j != 0 and j != img.shape[1]-1: # 最後一排、非左下非右下
                mask = [0, 1, 0, 1, -4, 1]
                neighbor = [img[i-1][j-1][0], img[i-1][j][0], img[i-1][j+1][0],
                            img[i][j-1][0], img[i][j][0], img[i][j+1][0]]
            elif i != 0 and i != img.shape[0]-1 and j == 0: #最左側 非左上飛左下
                mask = [1, 0, -4, 1, 1, 0]
                neighbor = [img[i-1][j][0], img[i-1][j+1][0],
                            img[i][j][0], img[i][j+1][0],
                            img[i+1][j][0], img[i+1][j+1][0]]
            elif i != 0 and i != img.shape[0]-1 and j == img.shape[1]-1: # 最右側 非右上非右下
                mask = [0, 1, 1, -4, 0, 1]
                neighbor = [img[i-1][j-1][0], img[i-1][j][0],
                            img[i][j-1][0], img[i][j][0],
                            img[i+1][j-1][0], img[i+1][j][0]]
            else:                
                mask = [0, 1, 0, 1, -4, 1, 0, 1, 0]
                neighbor = [img[i-1][j-1][0], img[i-1][j][0], img[i-1][j+1][0],
                            img[i][j-1][0], img[i][j][0], img[i][j+1][0],
                            img[i+1][j-1][0], img[i+1][j][0], img[i+1][j+1][0]]
                
            ipgm = sum([mask[tmp] * neighbor[tmp] for tmp in range(len(mask))])
            
            if ipgm >= threshold:
                lap_map[i][j] = 1
            elif ipgm <= -threshold:
                lap_map[i][j] = -1
            else:
                lap_map[i][j] = 0
                
    for i in range(lap_map.shape[0]):
        for j in range(lap_map.shape[1]):
            if lap_map[i][j] == 1:
                if i == 0 and j == 0:
                    neighbor = [lap_map[i][j], lap_map[i][j+1], lap_map[i+1][j], lap_map[i+1][j+1]]                    
                elif i == 0 and j == img.shape[1]-1:
                    neighbor = [lap_map[i][j-1], lap_map[i][j], lap_map[i+1][j-1], lap_map[i+1][j]]                
                elif i == 0 and j != 0 and j != img.shape[1]-1:
                    neighbor = [lap_map[i][j-1], lap_map[i][j], lap_map[i][j+1],
                                lap_map[i+1][j-1],lap_map[i+1][j], lap_map[i+1][j+1]]                
                elif i == img.shape[0]-1 and j == 0:
                    neighbor = [lap_map[i-1][j], lap_map[i-1][j+1], lap_map[i][j], lap_map[i][j+1]]                   
                elif i == img.shape[0]-1 and j == img.shape[1]-1:
                    neighbor = [lap_map[i-1][j-1], lap_map[i-1][j], lap_map[i][j-1], lap_map[i][j]]    
                elif i == img.shape[0]-1 and j != 0 and j != img.shape[1]-1:
                    neighbor = [lap_map[i-1][j-1], lap_map[i-1][j], lap_map[i-1][j+1],
                                lap_map[i][j-1], lap_map[i][j], lap_map[i][j+1]]
                elif i != 0 and i != img.shape[0]-1 and j == 0:
                    neighbor = [lap_map[i-1][j], lap_map[i-1][j+1],lap_map[i][j],
                                lap_map[i][j+1], lap_map[i+1][j], lap_map[i+1][j+1]]
                elif i != 0 and i != img.shape[0]-1 and j == img.shape[1]-1:
                    neighbor = [lap_map[i-1][j-1], lap_map[i-1][j],lap_map[i][j-1], 
                                lap_map[i][j],lap_map[i+1][j-1], lap_map[i+1][j]]
                else:
                    neighbor = [lap_map[i-1][j-1], lap_map[i-1][j], lap_map[i-1][j+1],
                                lap_map[i][j-1], lap_map[i][j], lap_map[i][j+1],
                                lap_map[i+1][j-1], lap_map[i+1][j], lap_map[i+1][j+1]]
                
                if -1 not in neighbor:
                    img_lap[i][j] = 255
            else:
                img_lap[i][j] = 255  
    
    return img_lap

def Lapalce2(img, threshold):
    img_lap = np.zeros((img.shape[0], img.shape[1]))
    lap_map = np.zeros((img.shape[0], img.shape[1]))
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i == 0 and j == 0: # 左上
                mask = [-8, 1, 1, 1]
                neighbor = [img[i][j][0], img[i][j+1][0],
                            img[i+1][j][0], img[i+1][j+1][0]]
            elif i == 0 and j == img.shape[1]-1: # 右上
                mask = [1, -8, 1, 1]
                neighbor = [img[i][j-1][0], img[i][j][0],
                            img[i+1][j-1][0], img[i+1][j][0]]
            elif i == 0 and j != 0 and j != img.shape[1]-1: #第一排、非左上飛右上
                mask = [1, -8, 1, 1, 1, 1]
                neighbor = [img[i][j-1][0], img[i][j][0], img[i][j+1][0],
                            img[i+1][j-1][0], img[i+1][j][0], img[i+1][j+1][0]]
            elif i == img.shape[0]-1 and j == 0: # 左下
                mask = [1, 1, -8, 1]
                neighbor = [img[i-1][j][0], img[i-1][j+1][0],
                            img[i][j][0], img[i][j+1][0]]
            elif i == img.shape[0]-1 and j == img.shape[1]-1: # 右下
                mask = [1, 1, 1, -8]
                neighbor = [img[i-1][j-1][0], img[i-1][j][0],
                            img[i][j-1][0], img[i][j][0]]
            elif i == img.shape[0]-1 and j != 0 and j != img.shape[1]-1: # 最後一排、非左下非右下
                mask = [1, 1, 1, 1, -8, 1]
                neighbor = [img[i-1][j-1][0], img[i-1][j][0], img[i-1][j+1][0],
                            img[i][j-1][0], img[i][j][0], img[i][j+1][0]]
            elif i != 0 and i != img.shape[0]-1 and j == 0: #最左側 非左上飛左下
                mask = [1, 1, -8, 1, 1, 1]
                neighbor = [img[i-1][j][0], img[i-1][j+1][0],
                            img[i][j][0], img[i][j+1][0],
                            img[i+1][j][0], img[i+1][j+1][0]]
            elif i != 0 and i != img.shape[0]-1 and j == img.shape[1]-1: # 最右側 非右上非右下
                mask = [1, 1, 1, -8, 1, 1]
                neighbor = [img[i-1][j-1][0], img[i-1][j][0],
                            img[i][j-1][0], img[i][j][0],
                            img[i+1][j-1][0], img[i+1][j][0]]
            else:                
                mask = [1, 1, 1, 1, -8, 1, 1, 1, 1]
                neighbor = [img[i-1][j-1][0], img[i-1][j][0], img[i-1][j+1][0],
                            img[i][j-1][0], img[i][j][0], img[i][j+1][0],
                            img[i+1][j-1][0], img[i+1][j][0], img[i+1][j+1][0]]
                
            ipgm = (1/3)*(sum([mask[tmp] * neighbor[tmp] for tmp in range(len(mask))]))
            
            if ipgm >= threshold:
                lap_map[i][j] = 1
            elif ipgm <= -threshold:
                lap_map[i][j] = -1
            else:
                lap_map[i][j] = 0
                
    for i in range(lap_map.shape[0]):
        for j in range(lap_map.shape[1]):
            if lap_map[i][j] == 1:
                if i == 0 and j == 0:
                    neighbor = [lap_map[i][j], lap_map[i][j+1], lap_map[i+1][j], lap_map[i+1][j+1]]                    
                elif i == 0 and j == img.shape[1]-1:
                    neighbor = [lap_map[i][j-1], lap_map[i][j], lap_map[i+1][j-1], lap_map[i+1][j]]                
                elif i == 0 and j != 0 and j != img.shape[1]-1:
                    neighbor = [lap_map[i][j-1], lap_map[i][j], lap_map[i][j+1],
                                lap_map[i+1][j-1],lap_map[i+1][j], lap_map[i+1][j+1]]                
                elif i == img.shape[0]-1 and j == 0:
                    neighbor = [lap_map[i-1][j], lap_map[i-1][j+1], lap_map[i][j], lap_map[i][j+1]]                   
                elif i == img.shape[0]-1 and j == img.shape[1]-1:
                    neighbor = [lap_map[i-1][j-1], lap_map[i-1][j], lap_map[i][j-1], lap_map[i][j]]    
                elif i == img.shape[0]-1 and j != 0 and j != img.shape[1]-1:
                    neighbor = [lap_map[i-1][j-1], lap_map[i-1][j], lap_map[i-1][j+1],
                                lap_map[i][j-1], lap_map[i][j], lap_map[i][j+1]]
                elif i != 0 and i != img.shape[0]-1 and j == 0:
                    neighbor = [lap_map[i-1][j], lap_map[i-1][j+1],lap_map[i][j],
                                lap_map[i][j+1], lap_map[i+1][j], lap_map[i+1][j+1]]
                elif i != 0 and i != img.shape[0]-1 and j == img.shape[1]-1:
                    neighbor = [lap_map[i-1][j-1], lap_map[i-1][j],lap_map[i][j-1], 
                                lap_map[i][j],lap_map[i+1][j-1], lap_map[i+1][j]]
                else:
                    neighbor = [lap_map[i-1][j-1], lap_map[i-1][j], lap_map[i-1][j+1],
                                lap_map[i][j-1], lap_map[i][j], lap_map[i][j+1],
                                lap_map[i+1][j-1], lap_map[i+1][j], lap_map[i+1][j+1]]
                
                if -1 not in neighbor:
                    img_lap[i][j] = 255
            else:
                img_lap[i][j] = 255  
    
    return img_lap

# minimum-variance Laplacian
def minvarLaplacian(img, threshold):
    img_lap = np.zeros((img.shape[0], img.shape[1]))
    lap_map = np.zeros((img.shape[0], img.shape[1]))
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i == 0 and j == 0: # 左上
                mask = [-4, -1, -1, 2]
                neighbor = [img[i][j][0], img[i][j+1][0],
                            img[i+1][j][0], img[i+1][j+1][0]]
            elif i == 0 and j == img.shape[1]-1: # 右上
                mask = [-1, -4, 2, -1]
                neighbor = [img[i][j-1][0], img[i][j][0],
                            img[i+1][j-1][0], img[i+1][j][0]]
            elif i == 0 and j != 0 and j != img.shape[1]-1: #第一排、非左上飛右上
                mask = [-1, -4, -1, 2, -1, 2]
                neighbor = [img[i][j-1][0], img[i][j][0], img[i][j+1][0],
                            img[i+1][j-1][0], img[i+1][j][0], img[i+1][j+1][0]]
            elif i == img.shape[0]-1 and j == 0: # 左下
                mask = [-1, 2, -4, -1]
                neighbor = [img[i-1][j][0], img[i-1][j+1][0],
                            img[i][j][0], img[i][j+1][0]]
            elif i == img.shape[0]-1 and j == img.shape[1]-1: # 右下
                mask = [2, -1, -1, -4]
                neighbor = [img[i-1][j-1][0], img[i-1][j][0],
                            img[i][j-1][0], img[i][j][0]]
            elif i == img.shape[0]-1 and j != 0 and j != img.shape[1]-1: # 最後一排、非左下非右下
                mask = [2, -1, 2, -1, -4, -1]
                neighbor = [img[i-1][j-1][0], img[i-1][j][0], img[i-1][j+1][0],
                            img[i][j-1][0], img[i][j][0], img[i][j+1][0]]
            elif i != 0 and i != img.shape[0]-1 and j == 0: #最左側 非左上飛左下
                mask = [-1, 2, -4, -1, -1, 2]
                neighbor = [img[i-1][j][0], img[i-1][j+1][0],
                            img[i][j][0], img[i][j+1][0],
                            img[i+1][j][0], img[i+1][j+1][0]]
            elif i != 0 and i != img.shape[0]-1 and j == img.shape[1]-1: # 最右側 非右上非右下
                mask = [2, -1, -1, -4, 2, -1]
                neighbor = [img[i-1][j-1][0], img[i-1][j][0],
                            img[i][j-1][0], img[i][j][0],
                            img[i+1][j-1][0], img[i+1][j][0]]
            else:                
                mask = [2, -1, 2, -1, -4, -1, 2, -1, 2]
                neighbor = [img[i-1][j-1][0], img[i-1][j][0], img[i-1][j+1][0],
                            img[i][j-1][0], img[i][j][0], img[i][j+1][0],
                            img[i+1][j-1][0], img[i+1][j][0], img[i+1][j+1][0]]
                
            ipgm = (1/3)*(sum([mask[tmp] * neighbor[tmp] for tmp in range(len(mask))]))
            
            if ipgm >= threshold:
                lap_map[i][j] = 1
            elif ipgm <= -threshold:
                lap_map[i][j] = -1
            else:
                lap_map[i][j] = 0
                
    for i in range(lap_map.shape[0]):
        for j in range(lap_map.shape[1]):
            if lap_map[i][j] == 1:
                if i == 0 and j == 0:
                    neighbor = [lap_map[i][j], lap_map[i][j+1], lap_map[i+1][j], lap_map[i+1][j+1]]                    
                elif i == 0 and j == img.shape[1]-1:
                    neighbor = [lap_map[i][j-1], lap_map[i][j], lap_map[i+1][j-1], lap_map[i+1][j]]                
                elif i == 0 and j != 0 and j != img.shape[1]-1:
                    neighbor = [lap_map[i][j-1], lap_map[i][j], lap_map[i][j+1],
                                lap_map[i+1][j-1],lap_map[i+1][j], lap_map[i+1][j+1]]                
                elif i == img.shape[0]-1 and j == 0:
                    neighbor = [lap_map[i-1][j], lap_map[i-1][j+1], lap_map[i][j], lap_map[i][j+1]]                   
                elif i == img.shape[0]-1 and j == img.shape[1]-1:
                    neighbor = [lap_map[i-1][j-1], lap_map[i-1][j], lap_map[i][j-1], lap_map[i][j]]    
                elif i == img.shape[0]-1 and j != 0 and j != img.shape[1]-1:
                    neighbor = [lap_map[i-1][j-1], lap_map[i-1][j], lap_map[i-1][j+1],
                                lap_map[i][j-1], lap_map[i][j], lap_map[i][j+1]]
                elif i != 0 and i != img.shape[0]-1 and j == 0:
                    neighbor = [lap_map[i-1][j], lap_map[i-1][j+1],lap_map[i][j],
                                lap_map[i][j+1], lap_map[i+1][j], lap_map[i+1][j+1]]
                elif i != 0 and i != img.shape[0]-1 and j == img.shape[1]-1:
                    neighbor = [lap_map[i-1][j-1], lap_map[i-1][j],lap_map[i][j-1], 
                                lap_map[i][j],lap_map[i+1][j-1], lap_map[i+1][j]]
                else:
                    neighbor = [lap_map[i-1][j-1], lap_map[i-1][j], lap_map[i-1][j+1],
                                lap_map[i][j-1], lap_map[i][j], lap_map[i][j+1],
                                lap_map[i+1][j-1], lap_map[i+1][j], lap_map[i+1][j+1]]
                
                if -1 not in neighbor:
                    img_lap[i][j] = 255
            else:
                img_lap[i][j] = 255  
    
    return img_lap    

# Laplacian of Gaussian
def LOG(img, threshold):
    img_padding = copy.deepcopy(img)
    img_padding = cv2.copyMakeBorder(img_padding,5,5,5,5,cv2.BORDER_CONSTANT,value=0)
    img_lap = np.zeros((img.shape[0], img.shape[1]))
    lap_map = np.zeros((img.shape[0], img.shape[1]))
    
    mask = [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0,
            0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0,
            0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0,
            -1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1,
            -1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1,
            -2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2,
            -1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1,
            -1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1,
            0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0,
            0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0,
            0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0]
    
    for i in range(5, img_padding.shape[0]-5):
        for j in range(5, img_padding.shape[1]-5):
            
            neighbor = [img_padding[i-5][j-5][0], img_padding[i-5][j-4][0], img_padding[i-5][j-3][0], img_padding[i-5][j-2][0],
                    img_padding[i-5][j-1][0], img_padding[i-5][j][0], img_padding[i-5][j+1][0], img_padding[i-5][j+2][0],
                    img_padding[i-5][j+3][0], img_padding[i-5][j+4][0], img_padding[i-5][j+5][0],
                    img_padding[i-4][j-5][0], img_padding[i-4][j-4][0], img_padding[i-4][j-3][0], img_padding[i-4][j-2][0],
                    img_padding[i-4][j-1][0], img_padding[i-4][j][0], img_padding[i-4][j+1][0], img_padding[i-4][j+2][0],
                    img_padding[i-4][j+3][0], img_padding[i-4][j+4][0], img_padding[i-4][j+5][0],
                    img_padding[i-3][j-5][0], img_padding[i-3][j-4][0], img_padding[i-3][j-3][0], img_padding[i-3][j-2][0],
                    img_padding[i-3][j-1][0], img_padding[i-3][j][0], img_padding[i-3][j+1][0], img_padding[i-3][j+2][0],
                    img_padding[i-3][j+3][0], img_padding[i-3][j+4][0], img_padding[i-3][j+5][0],
                    img_padding[i-2][j-5][0], img_padding[i-2][j-4][0], img_padding[i-2][j-3][0], img_padding[i-2][j-2][0],
                    img_padding[i-2][j-1][0], img_padding[i-2][j][0], img_padding[i-2][j+1][0], img_padding[i-2][j+2][0],
                    img_padding[i-2][j+3][0], img_padding[i-2][j+4][0], img_padding[i-2][j+5][0],
                    img_padding[i-1][j-5][0], img_padding[i-1][j-4][0], img_padding[i-1][j-3][0], img_padding[i-5][j-2][0],
                    img_padding[i-1][j-1][0], img_padding[i-1][j][0], img_padding[i-1][j+1][0], img_padding[i-5][j+2][0],
                    img_padding[i-1][j+3][0], img_padding[i-1][j+4][0], img_padding[i-1][j+5][0],
                    img_padding[i][j-5][0], img_padding[i][j-4][0], img_padding[i][j-3][0], img_padding[i][j-2][0],
                    img_padding[i][j-1][0], img_padding[i][j][0], img_padding[i][j+1][0], img_padding[i][j+2][0],
                    img_padding[i][j+3][0], img_padding[i][j+4][0], img_padding[i][j+5][0],
                    img_padding[i+1][j-5][0], img_padding[i+1][j-4][0], img_padding[i+1][j-3][0], img_padding[i+1][j-2][0],
                    img_padding[i+1][j-1][0], img_padding[i+1][j][0], img_padding[i+1][j+1][0], img_padding[i+1][j+2][0],
                    img_padding[i+1][j+3][0], img_padding[i+1][j+4][0], img_padding[i+1][j+5][0],
                    img_padding[i+2][j-5][0], img_padding[i+2][j-4][0], img_padding[i+2][j-3][0], img_padding[i+2][j-2][0],
                    img_padding[i+2][j-1][0], img_padding[i+2][j][0], img_padding[i+2][j+1][0], img_padding[i+2][j+2][0],
                    img_padding[i+2][j+3][0], img_padding[i+2][j+4][0], img_padding[i+2][j+5][0],
                    img_padding[i+3][j-5][0], img_padding[i+3][j-4][0], img_padding[i+3][j-3][0], img_padding[i+3][j-2][0],
                    img_padding[i+3][j-1][0], img_padding[i+3][j][0], img_padding[i+3][j+1][0], img_padding[i+3][j+2][0],
                    img_padding[i+3][j+3][0], img_padding[i+3][j+4][0], img_padding[i+3][j+5][0],
                    img_padding[i+4][j-5][0], img_padding[i+4][j-4][0], img_padding[i+4][j-3][0], img_padding[i+4][j-2][0],
                    img_padding[i+4][j-1][0], img_padding[i+4][j][0], img_padding[i+4][j+1][0], img_padding[i+4][j+2][0],
                    img_padding[i+4][j+3][0], img_padding[i+4][j+4][0], img_padding[i+4][j+5][0],
                    img_padding[i+5][j-5][0], img_padding[i+5][j-4][0], img_padding[i+5][j-3][0], img_padding[i+5][j-2][0],
                    img_padding[i+5][j-1][0], img_padding[i+5][j][0], img_padding[i+5][j+1][0], img_padding[i+5][j+2][0],
                    img_padding[i+5][j+3][0], img_padding[i+5][j+4][0], img_padding[i+5][j+5][0]]            
            
            ipgm = sum([mask[tmp] * neighbor[tmp] for tmp in range(len(mask))])
            
            if ipgm >= threshold:
                lap_map[i-5][j-5] = 1
            elif ipgm <= -threshold:
                lap_map[i-5][j-5] = -1
            else:
                lap_map[i-5][j-5] = 0

    for i in range(lap_map.shape[0]):
        for j in range(lap_map.shape[1]):
            if lap_map[i][j] == 1:
                if i == 0 and j == 0:
                    neighbor = [lap_map[i][j], lap_map[i][j+1], lap_map[i+1][j], lap_map[i+1][j+1]]                    
                elif i == 0 and j == img.shape[1]-1:
                    neighbor = [lap_map[i][j-1], lap_map[i][j], lap_map[i+1][j-1], lap_map[i+1][j]]                
                elif i == 0 and j != 0 and j != img.shape[1]-1:
                    neighbor = [lap_map[i][j-1], lap_map[i][j], lap_map[i][j+1],
                                lap_map[i+1][j-1],lap_map[i+1][j], lap_map[i+1][j+1]]                
                elif i == img.shape[0]-1 and j == 0:
                    neighbor = [lap_map[i-1][j], lap_map[i-1][j+1], lap_map[i][j], lap_map[i][j+1]]                   
                elif i == img.shape[0]-1 and j == img.shape[1]-1:
                    neighbor = [lap_map[i-1][j-1], lap_map[i-1][j], lap_map[i][j-1], lap_map[i][j]]    
                elif i == img.shape[0]-1 and j != 0 and j != img.shape[1]-1:
                    neighbor = [lap_map[i-1][j-1], lap_map[i-1][j], lap_map[i-1][j+1],
                                lap_map[i][j-1], lap_map[i][j], lap_map[i][j+1]]
                elif i != 0 and i != img.shape[0]-1 and j == 0:
                    neighbor = [lap_map[i-1][j], lap_map[i-1][j+1],lap_map[i][j],
                                lap_map[i][j+1], lap_map[i+1][j], lap_map[i+1][j+1]]
                elif i != 0 and i != img.shape[0]-1 and j == img.shape[1]-1:
                    neighbor = [lap_map[i-1][j-1], lap_map[i-1][j],lap_map[i][j-1], 
                                lap_map[i][j],lap_map[i+1][j-1], lap_map[i+1][j]]
                else:
                    neighbor = [lap_map[i-1][j-1], lap_map[i-1][j], lap_map[i-1][j+1],
                                lap_map[i][j-1], lap_map[i][j], lap_map[i][j+1],
                                lap_map[i+1][j-1], lap_map[i+1][j], lap_map[i+1][j+1]]
                
                if -1 not in neighbor:
                    img_lap[i][j] = 255
            else:
                img_lap[i][j] = 255  
    
    return img_lap

# Difference of Gaussian
def DOG(img, threshold):
    img_padding = copy.deepcopy(img)
    img_padding = cv2.copyMakeBorder(img_padding,5,5,5,5,cv2.BORDER_CONSTANT,value=0)
    img_gau = np.zeros((img.shape[0], img.shape[1]))
    gau_map = np.zeros((img.shape[0], img.shape[1]))
    
    mask = [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1,
            -3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3,
            -4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4,
            -6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6,
            -7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7,
            -8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8,
            -7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7,
            -6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6,
            -4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4,
            -3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3,
            -1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1]
    
    for i in range(5, img_padding.shape[0]-5):
        for j in range(5, img_padding.shape[1]-5):
            
            neighbor = [img_padding[i-5][j-5][0], img_padding[i-5][j-4][0], img_padding[i-5][j-3][0], img_padding[i-5][j-2][0],
                    img_padding[i-5][j-1][0], img_padding[i-5][j][0], img_padding[i-5][j+1][0], img_padding[i-5][j+2][0],
                    img_padding[i-5][j+3][0], img_padding[i-5][j+4][0], img_padding[i-5][j+5][0],
                    img_padding[i-4][j-5][0], img_padding[i-4][j-4][0], img_padding[i-4][j-3][0], img_padding[i-4][j-2][0],
                    img_padding[i-4][j-1][0], img_padding[i-4][j][0], img_padding[i-4][j+1][0], img_padding[i-4][j+2][0],
                    img_padding[i-4][j+3][0], img_padding[i-4][j+4][0], img_padding[i-4][j+5][0],
                    img_padding[i-3][j-5][0], img_padding[i-3][j-4][0], img_padding[i-3][j-3][0], img_padding[i-3][j-2][0],
                    img_padding[i-3][j-1][0], img_padding[i-3][j][0], img_padding[i-3][j+1][0], img_padding[i-3][j+2][0],
                    img_padding[i-3][j+3][0], img_padding[i-3][j+4][0], img_padding[i-3][j+5][0],
                    img_padding[i-2][j-5][0], img_padding[i-2][j-4][0], img_padding[i-2][j-3][0], img_padding[i-2][j-2][0],
                    img_padding[i-2][j-1][0], img_padding[i-2][j][0], img_padding[i-2][j+1][0], img_padding[i-2][j+2][0],
                    img_padding[i-2][j+3][0], img_padding[i-2][j+4][0], img_padding[i-2][j+5][0],
                    img_padding[i-1][j-5][0], img_padding[i-1][j-4][0], img_padding[i-1][j-3][0], img_padding[i-5][j-2][0],
                    img_padding[i-1][j-1][0], img_padding[i-1][j][0], img_padding[i-1][j+1][0], img_padding[i-5][j+2][0],
                    img_padding[i-1][j+3][0], img_padding[i-1][j+4][0], img_padding[i-1][j+5][0],
                    img_padding[i][j-5][0], img_padding[i][j-4][0], img_padding[i][j-3][0], img_padding[i][j-2][0],
                    img_padding[i][j-1][0], img_padding[i][j][0], img_padding[i][j+1][0], img_padding[i][j+2][0],
                    img_padding[i][j+3][0], img_padding[i][j+4][0], img_padding[i][j+5][0],
                    img_padding[i+1][j-5][0], img_padding[i+1][j-4][0], img_padding[i+1][j-3][0], img_padding[i+1][j-2][0],
                    img_padding[i+1][j-1][0], img_padding[i+1][j][0], img_padding[i+1][j+1][0], img_padding[i+1][j+2][0],
                    img_padding[i+1][j+3][0], img_padding[i+1][j+4][0], img_padding[i+1][j+5][0],
                    img_padding[i+2][j-5][0], img_padding[i+2][j-4][0], img_padding[i+2][j-3][0], img_padding[i+2][j-2][0],
                    img_padding[i+2][j-1][0], img_padding[i+2][j][0], img_padding[i+2][j+1][0], img_padding[i+2][j+2][0],
                    img_padding[i+2][j+3][0], img_padding[i+2][j+4][0], img_padding[i+2][j+5][0],
                    img_padding[i+3][j-5][0], img_padding[i+3][j-4][0], img_padding[i+3][j-3][0], img_padding[i+3][j-2][0],
                    img_padding[i+3][j-1][0], img_padding[i+3][j][0], img_padding[i+3][j+1][0], img_padding[i+3][j+2][0],
                    img_padding[i+3][j+3][0], img_padding[i+3][j+4][0], img_padding[i+3][j+5][0],
                    img_padding[i+4][j-5][0], img_padding[i+4][j-4][0], img_padding[i+4][j-3][0], img_padding[i+4][j-2][0],
                    img_padding[i+4][j-1][0], img_padding[i+4][j][0], img_padding[i+4][j+1][0], img_padding[i+4][j+2][0],
                    img_padding[i+4][j+3][0], img_padding[i+4][j+4][0], img_padding[i+4][j+5][0],
                    img_padding[i+5][j-5][0], img_padding[i+5][j-4][0], img_padding[i+5][j-3][0], img_padding[i+5][j-2][0],
                    img_padding[i+5][j-1][0], img_padding[i+5][j][0], img_padding[i+5][j+1][0], img_padding[i+5][j+2][0],
                    img_padding[i+5][j+3][0], img_padding[i+5][j+4][0], img_padding[i+5][j+5][0]]            
            
            ipgm = sum([mask[tmp] * neighbor[tmp] for tmp in range(len(mask))])
            
            if ipgm >= threshold:
                gau_map[i-5][j-5] = 1
            elif ipgm <= -threshold:
                gau_map[i-5][j-5] = -1
            else:
                gau_map[i-5][j-5] = 0

    for i in range(gau_map.shape[0]):
        for j in range(gau_map.shape[1]):
            if gau_map[i][j] == 1:
                if i == 0 and j == 0:
                    neighbor = [gau_map[i][j], gau_map[i][j+1], gau_map[i+1][j], gau_map[i+1][j+1]]                    
                elif i == 0 and j == img.shape[1]-1:
                    neighbor = [gau_map[i][j-1], gau_map[i][j], gau_map[i+1][j-1], gau_map[i+1][j]]                
                elif i == 0 and j != 0 and j != img.shape[1]-1:
                    neighbor = [gau_map[i][j-1], gau_map[i][j], gau_map[i][j+1],
                                gau_map[i+1][j-1],gau_map[i+1][j], gau_map[i+1][j+1]]                
                elif i == img.shape[0]-1 and j == 0:
                    neighbor = [gau_map[i-1][j], gau_map[i-1][j+1], gau_map[i][j], gau_map[i][j+1]]                   
                elif i == img.shape[0]-1 and j == img.shape[1]-1:
                    neighbor = [gau_map[i-1][j-1], gau_map[i-1][j], gau_map[i][j-1], gau_map[i][j]]    
                elif i == img.shape[0]-1 and j != 0 and j != img.shape[1]-1:
                    neighbor = [gau_map[i-1][j-1], gau_map[i-1][j], gau_map[i-1][j+1],
                                gau_map[i][j-1], gau_map[i][j], gau_map[i][j+1]]
                elif i != 0 and i != img.shape[0]-1 and j == 0:
                    neighbor = [gau_map[i-1][j], gau_map[i-1][j+1],gau_map[i][j],
                                gau_map[i][j+1], gau_map[i+1][j], gau_map[i+1][j+1]]
                elif i != 0 and i != img.shape[0]-1 and j == img.shape[1]-1:
                    neighbor = [gau_map[i-1][j-1], gau_map[i-1][j],gau_map[i][j-1], 
                                gau_map[i][j],gau_map[i+1][j-1], gau_map[i+1][j]]
                else:
                    neighbor = [gau_map[i-1][j-1], gau_map[i-1][j], gau_map[i-1][j+1],
                                gau_map[i][j-1], gau_map[i][j], gau_map[i][j+1],
                                gau_map[i+1][j-1], gau_map[i+1][j], gau_map[i+1][j+1]]
                
                if -1 not in neighbor:
                    img_gau[i][j] = 255
            else:
                img_gau[i][j] = 255  
    
    return img_gau

img = cv2.imread('lena.bmp')
cv2.imwrite('img_lap1_15.bmp',Lapalce1(img, 15))
cv2.imwrite('img_lap2_15.bmp',Lapalce2(img, 15))
cv2.imwrite('img_minvarlap20.bmp',minvarLaplacian(img, 20))
cv2.imwrite('img_log3000.bmp',LOG(img, 3000))
cv2.imwrite('img_gau1.bmp',DOG(img, 1))