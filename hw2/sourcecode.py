import matplotlib.pyplot as plt
import numpy as np
import cv2
import itertools
from collections import Counter

#讀取檔案（numpy array）
img = cv2.imread('lena.bmp')

#Binariaze image at threshold
#建立輸出的圖片檔
cv2.imwrite('binarize.bmp', img)
#開啟圖片
img1 = cv2.imread('binarize.bmp')
for i in range(512):
    for j in range(512):
        if int(img1[i][j][0]) >= 128:
            img1[i][j] = [255, 255, 255]
        else:
            img1[i][j] = [0, 0, 0]
cv2.imwrite('binarize.bmp', img1)

#Histogram
def draw_hist(myList,Title,Xlabel,Ylabel):
    plt.hist(myList, 50)
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.title(Title)
    plt.savefig("Histogram.png")
    plt.show()

data = [] #用list儲存數值
for i in range(512):
    for j in range(512):
        data.append(int(img[i][j][0]))    
draw_hist(data,'Gray Level Histogram','Gray level value','number')


#Connected components

labelArray = [[0 for _ in range(512)] for _ in range(512)]

#initialize each pixel to a new label
newlabel = 1
for i in range(512):
    for j in range(512):
        if int(img1[i][j][0]) != 0:
            labelArray[i][j] = newlabel
            newlabel += 1

#Iteration of top-down followed by bottom-up
#使用四連通
while True:
    change = False
    #top-down pass
    for i in range(512):
        for j in range(512):
            if labelArray[i][j] != 0:
                u = 2700000
                l = 2700000
                #可以往上
                if (i != 0) and (labelArray[i-1][j] != 0):
                    u = labelArray[i-1][j]
                #可以往左
                if (j != 0) and (labelArray[i][j-1] != 0):
                    l = labelArray[i][j-1]
                m = min(u, l)
                if (m != 0) and (m < labelArray[i][j]):
                    change = True
                    labelArray[i][j] = m
                    
    #bottom-up pass
    for i in range(511, -1, -1):
        for j in range(511, -1, -1):
            if labelArray[i][j] != 0:
                d = 2700000
                r = 2700000
                #可以往下
                if (i != 511) and (labelArray[i+1][j] != 0):
                    d = labelArray[i+1][j]
                #可以往右
                if (j != 511) and (labelArray[i][j+1] != 0):
                    r = labelArray[i][j+1]
                m = min(d, r)
                if(m != 0) and (m < labelArray[i][j]):
                    change = True
                    labelArray[i][j] = m
    if change == False:
        break

# count the number of regions ...
test = list(itertools.chain(*labelArray))
sorted_dict =  Counter(test)
a = sorted(sorted_dict.items(), key=lambda x: x[1])

label_List = []
for i in range(len(a)-1, -1, -1):
    if(a[i][1] >= 500)and(a[i][0]!=0):
        label_List.append(a[i][0])


#draw bounding box and centroid...
for l in range(len(label_List)):
    updown = []
    leftright = []

    for i in range(512):
        for j in range(512):
            if labelArray[i][j] == label_List[l]:
                updown.append(i)
                leftright.append(j)
                
    x = int(sum(leftright)/len(leftright))
    y = int(sum(updown)/len(updown))
    cv2.rectangle(img1, (min(leftright), min(updown)), (max(leftright), max(updown)), (255, 0, 0), 1)
    cv2.circle(img1,(x, y), 4, (0, 0, 255), -1)
    
cv2.imwrite('connect_component.bmp', img1)