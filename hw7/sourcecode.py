import numpy as np
import copy
import cv2

def binarize(img, threshold):
    for i in range(512):
        for j in range(512):
            if int(img[i][j][0]) >= threshold:
                img[i][j] = [255, 255, 255]
            else:
                img[i][j] = [0, 0, 0]
    return img

def compute_yokoi_number(img_org):
    
    def h(b, c, d, e):
        if b == c and (d != b or e != b):
            return 'q'
        if b == c and (d == b and e == b):
            return 'r'
        return 's'

    yokoi = np.zeros(img_org.shape)
    
    for i in range(img_org.shape[0]):
        for j in range(img_org.shape[1]):
            if img_org[i][j] > 0:  # not backgroung pixel
                if i == 0:
                    if j == 0:
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = 0, img_org[i][j], img_org[i][j + 1]
                        x8, x4, x5 = 0, img_org[i + 1][j], img_org[i + 1][j + 1]
                    elif j == img_org.shape[1] - 1:
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = img_org[i][j - 1], img_org[i][j], 0
                        x8, x4, x5 = img_org[i + 1][j - 1], img_org[i + 1][j], 0
                    else:
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = img_org[i][j - 1], img_org[i][j], img_org[i][j + 1]
                        x8, x4, x5 = img_org[i + 1][j - 1], img_org[i + 1][j], img_org[i + 1][j + 1]
                elif i == img_org.shape[0] - 1:
                    if j == 0:
                        x7, x2, x6 = 0, img_org[i - 1][j], img_org[i - 1][j + 1]
                        x3, x0, x1 = 0, img_org[i][j], img_org[i][j + 1]
                        x8, x4, x5 = 0, 0, 0
                    elif j == img_org.shape[1] - 1:
                        x7, x2, x6 = img_org[i - 1][j - 1], img_org[i - 1][j], 0
                        x3, x0, x1 = img_org[i][j - 1], img_org[i][j], 0
                        x8, x4, x5 = 0, 0, 0
                    else:
                        x7, x2, x6 = img_org[i - 1][j - 1], img_org[i - 1][j], img_org[i - 1][j + 1]
                        x3, x0, x1 = img_org[i][j - 1], img_org[i][j], img_org[i][j + 1]
                        x8, x4, x5 = 0, 0, 0
                else:
                    if j == 0:
                        x7, x2, x6 = 0, img_org[i - 1][j], img_org[i - 1][j + 1]
                        x3, x0, x1 = 0, img_org[i][j], img_org[i][j + 1]
                        x8, x4, x5 = 0, img_org[i + 1][j], img_org[i + 1][j + 1]
                    elif j == img_org.shape[1] - 1:
                        x7, x2, x6 = img_org[i - 1][j - 1], img_org[i - 1][j], 0
                        x3, x0, x1 = img_org[i][j - 1], img_org[i][j], 0
                        x8, x4, x5 = img_org[i + 1][j - 1], img_org[i + 1][j], 0
                    else:
                        x7, x2, x6 = img_org[i - 1][j - 1], img_org[i - 1][j], img_org[i - 1][j + 1]
                        x3, x0, x1 = img_org[i][j - 1], img_org[i][j], img_org[i][j + 1]
                        x8, x4, x5 = img_org[i + 1][j - 1], img_org[i + 1][j], img_org[i + 1][j + 1]

                a1 = h(x0, x1, x6, x2)
                a2 = h(x0, x2, x7, x3)
                a3 = h(x0, x3, x8, x4)
                a4 = h(x0, x4, x5, x1)

                if a1 == 'r' and a2 == 'r' and a3 == 'r' and a4 == 'r':
                    ans = 5
                else:
                    ans = 0
                    for a_i in [a1, a2, a3, a4]:
                        if a_i == 'q':
                            ans += 1
                yokoi[i][j] = ans
    return yokoi

def mark_pair_relationship(img_ib):
    
    def h(a, m):
        if a == m:
            return 1
        return 0
    
    img_pair = np.zeros(img_ib.shape)
    for i in range(img_ib.shape[0]):
        for j in range(img_ib.shape[1]):
            if img_ib[i][j] > 0:
                # not background pixel
                x1, x2, x3, x4 = 0, 0, 0, 0
                if i == 0:
                    if j == 0:
                        x1, x4 = img_ib[i][j + 1], img_ib[i + 1][j]
                    elif j == img_ib.shape[1] - 1:
                        x3, x4 = img_ib[i][j - 1], img_ib[i + 1][j]
                    else:
                        x1, x3, x4 = img_ib[i][j + 1], img_ib[i][j - 1], img_ib[i + 1][j]
                elif i == img_ib.shape[0] - 1:
                    if j == 0:
                        x1, x2 = img_ib[i][j + 1], img_ib[i - 1][j]
                    elif j == img_ib.shape[1] - 1:
                        x2, x3 = img_ib[i - 1][j], img_ib[i][j - 1]
                    else:
                        x1, x2, x3 = img_ib[i][j + 1], img_ib[i - 1][j], img_ib[i][j - 1]
                else:
                    if j == 0:
                        x1, x2, x4 = img_ib[i][j + 1], img_ib[i - 1][j], img_ib[i + 1][j]
                    elif j == img_ib.shape[1] - 1:
                        x2, x3, x4 = img_ib[i - 1][j], img_ib[i][j - 1], img_ib[i + 1][j]
                    else:
                        x1, x2, x3, x4 = img_ib[i][j + 1], img_ib[i - 1][j], img_ib[i][j - 1], img_ib[i + 1][j]
                if h(x1, 1) + h(x2, 1) + h(x3, 1) + h(x4, 1) >= 1 and img_ib[i][j] == 1:
                    img_pair[i][j] = 1 # p
                else:
                    img_pair[i][j] = 2 # q
    return img_pair

# open, binarize, downsample lena.bmp
img = cv2.imread('lena.bmp')
img = binarize(img, 128)

p = np.zeros((64, 64))
for i in range(p.shape[0]):
    for j in range(p.shape[1]):
        p[i][j] = img[8 * i][8 * j][0]

# Thinning......
p_thin = copy.deepcopy(p)

while True:
    
    p_thin_old = copy.deepcopy(p_thin)

    # Step 1:  Yokoi Operator
    p_yokoi = compute_yokoi_number(p_thin_old)

    # Step 2:  Pair Relationship Operator
    p_mark = mark_pair_relationship(p_yokoi)

    # Step 3:  Connected Shrink Operator
    
    def h(b, c, d, e):
        if b == c and (d != b or e != b):
            return 1
        else:
            return 0
    
    for i in range(p_mark.shape[0]):
        for j in range(p_mark.shape[1]):
            if p_mark[i][j] == 1: # p
                if i == 0:
                    if j == 0:
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = 0, p_thin[i][j], p_thin[i][j + 1]
                        x8, x4, x5 = 0, p_thin[i + 1][j], p_thin[i + 1][j + 1]
                    elif j == p_thin.shape[1] - 1:
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = p_thin[i][j - 1], p_thin[i][j], 0
                        x8, x4, x5 = p_thin[i + 1][j - 1], p_thin[i + 1][j], 0
                    else:
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = p_thin[i][j - 1], p_thin[i][j], p_thin[i][j + 1]
                        x8, x4, x5 = p_thin[i + 1][j - 1], p_thin[i + 1][j], p_thin[i + 1][j + 1]
                elif i == p_mark.shape[0] - 1:
                    if j == 0:
                        x7, x2, x6 = 0, p_thin[i - 1][j], p_thin[i - 1][j + 1]
                        x3, x0, x1 = 0, p_thin[i][j], p_thin[i][j + 1]
                        x8, x4, x5 = 0, 0, 0
                    elif j == p_mark.shape[1] - 1:
                        x7, x2, x6 = p_thin[i - 1][j - 1], p_thin[i - 1][j], 0
                        x3, x0, x1 = p_thin[i][j - 1], p_thin[i][j], 0
                        x8, x4, x5 = 0, 0, 0
                    else:
                        x7, x2, x6 = p_thin[i - 1][j - 1], p_thin[i - 1][j], p_thin[i - 1][j + 1]
                        x3, x0, x1 = p_thin[i][j - 1], p_thin[i][j], p_thin[i][j + 1]
                        x8, x4, x5 = 0, 0, 0
                else:
                    if j == 0:
                        x7, x2, x6 = 0, p_thin[i - 1][j], p_thin[i - 1][j + 1]
                        x3, x0, x1 = 0, p_thin[i][j], p_thin[i][j + 1]
                        x8, x4, x5 = 0, p_thin[i + 1][j], p_thin[i + 1][j + 1]
                    elif j == p_mark.shape[1] - 1:
                        x7, x2, x6 = p_thin[i - 1][j - 1], p_thin[i - 1][j], 0
                        x3, x0, x1 = p_thin[i][j - 1], p_thin[i][j], 0
                        x8, x4, x5 = p_thin[i + 1][j - 1], p_thin[i + 1][j], 0
                    else:
                        x7, x2, x6 = p_thin[i - 1][j - 1], p_thin[i - 1][j], p_thin[i - 1][j + 1]
                        x3, x0, x1 = p_thin[i][j - 1], p_thin[i][j], p_thin[i][j + 1]
                        x8, x4, x5 = p_thin[i + 1][j - 1], p_thin[i + 1][j], p_thin[i + 1][j + 1]

                a1 = h(x0, x1, x6, x2)
                a2 = h(x0, x2, x7, x3)
                a3 = h(x0, x3, x8, x4)
                a4 = h(x0, x4, x5, x1)

                if (a1 + a2 + a3 + a4) == 1:
                    p_thin[i][j] = 0
                    
    # ckeck if the last output stops changing
    if np.sum(p_thin == p_thin_old) == p_thin.shape[0] * p_thin.shape[1]:
        break

cv2.imwrite('lena.thinned.bmp', p_thin)