import numpy as np
import cv2

#讀取檔案（numpy array）
img = cv2.imread('lena.bmp')
#img.shape (512, 512, 3)

# Part1.

# (a) upside-down lena.bmp

#建立輸出的圖片檔
cv2.imwrite('upside-down lena.bmp', img)
#開啟圖片
img1 = cv2.imread('upside-down lena.bmp')

for i in range(512):
    img1[i] = img[511-i]
cv2.imwrite('upside-down lena.bmp', img1)


# (b) right-side-left lena.bmp

#建立輸出的圖片檔
cv2.imwrite('right-side-left lena.bmp', img)
#開啟圖片
img2 = cv2.imread('right-side-left lena.bmp')

for i in range(512):
    for j in range(512):
        img2[i][j] = img[i][511-j]
cv2.imwrite('right-side-left lena.bmp', img2)


# (c) diagonally flip lena.bmp

#建立輸出的圖片檔
cv2.imwrite('diagonally-flip-lena.bmp', img)
#開啟圖片
img3 = cv2.imread('diagonally-flip-lena.bmp')

for i in range(512):
    for j in range(512):
        img3[i][j] = img[511-i][511-j]
cv2.imwrite('diagonally-flip-lena.bmp', img3)


# Part2.

# (f) binarize lena.bmp at 128 to get a binary image

#讀取檔案（numpy array）
img = cv2.imread('lena.bmp')

#建立輸出的圖片檔
cv2.imwrite('binarize-at-128-lena.bmp', img)
#開啟圖片
img4 = cv2.imread('binarize-at-128-lena.bmp')

for i in range(512):
    for j in range(512):
        if int(img4[i][j][0]) >= 128:
            img4[i][j] = [255, 255, 255]
        else:
            img4[i][j] = [0, 0, 0]
cv2.imwrite('binarize-at-128-lena.bmp', img4)

