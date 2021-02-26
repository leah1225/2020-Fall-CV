import numpy as np
import cv2
import matplotlib.pyplot as plt

#Histogram
def draw_hist(myList,Title,Xlabel,Ylabel):
    plt.hist(myList, 40)
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.title(Title)
    plt.xlim(xmin = -10)
    plt.xlim(xmax = 300)


#a.original image and its histogram
img = cv2.imread('lena.bmp')
data = []
for i in range(512):
    for j in range(512):
        data.append(int(img[i][j][0]))    
hist = draw_hist(data,'Gray Level Histogram','Gray level value','number')
plt.savefig("original.png")


#b.image with intensity divided by 3 and its histogram
img2 = img/3
data2 = []
for i in range(512):
    for j in range(512):
        n = int(img2[i][j][0])
        data2.append(n)
        img2[i][j] = [n, n, n] 
data2 = np.array(data2)
draw_hist(data2,'Gray Level Histogram','Gray level value','number')
plt.savefig("divided-by-3.png")
cv2.imwrite('intensity-divided-by-3.bmp', img2)


#c.image after applying histogram equalization to (b) and its histogram
n = img.shape[0]*img.shape[1] #total pixel number
gl_list = np.unique(data2) #all unique gray level value
eq_list = np.zeros(len(gl_list)) #new gray level value

cumul = 0
for i in range(len(gl_list)):
    num = len(data2[data2==gl_list[i]])
    cumul += num
    eq_list[i] = 255.*cumul/n

for i in range(len(gl_list)):
    img2 = np.where(img2 == gl_list[i], eq_list[i], img2)
cv2.imwrite('after-equalization.bmp', img2)

data3 = []
for i in range(512):
    for j in range(512):
        data3.append(img2[i][j][0])
draw_hist(data3,'Gray Level Histogram','Gray level value','number')
plt.savefig("after-equalization.png")
