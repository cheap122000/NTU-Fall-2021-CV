import cv2
import numpy as np
import imageio

def CopyImg(img):
    h, w = img.shape
    ret = np.zeros((h, w), np.uint8)
    for i in range(h):
        for j in range(w):
            ret[i][j] = img[i][j]     
    return ret

def BinarizeImageAt128(img):
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if img[i][j] >= 128:
                img[i][j] = 255
            else:
                img[i][j] = 0
    return img

def resizeWith8x8(img):
    h, w = img.shape
    ret = np.zeros((int(h/8), int(w/8)), np.uint8)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            ret[int(i/8)][int(j/8)] = img[i][j]

    return ret

def YokoiConnectivityNumber(img):
    def func_h(b, c, d, e):
        if b == c and (d != b or e != b): 
            return 'q'
        if b == c and (d == b and e == b): 
            return 'r'
        return 's'

    h, w = img.shape
    ret = np.zeros((h, w), np.uint8)
    for i in range(h):
        for j in range(w):
            if img[i][j] > 0:
                x7, x2, x6 = (0 if i-1 < 0 or j-1 < 0 else 0 if img[i-1][j-1] == 0 else 1), (0 if i-1 < 0 else 0 if img[i-1][j] == 0 else 1), (0 if i-1 < 0 or j+1 == w else 0 if img[i-1][j+1] == 0 else 1)
                x3, x0, x1 = (0 if j-1 < 0 else 0 if img[i][j-1] == 0 else 1), (0 if img[i][j] == 0 else 1), (0 if j+1 == w else 0 if img[i][j+1] == 0 else 1)
                x8, x4, x5 = (0 if i+1 == h or j-1 < 0 else 0 if img[i+1][j-1] == 0 else 1), (0 if i+1 == h else 0 if img[i+1][j] == 0 else 1), (0 if i+1 == h or j+1 == w else 0 if img[i+1][j+1] == 0 else 1)

                a1 = func_h(x0, x1, x6, x2)
                a2 = func_h(x0, x2, x7, x3)
                a3 = func_h(x0, x3, x8, x4)
                a4 = func_h(x0, x4, x5, x1)

                if [a1, a2, a3, a4] == ['r', 'r', 'r', 'r']:
                    ret[i][j] = 5
                else:
                    ret[i][j] = 0
                    for a in [a1, a2, a3, a4]:
                        if a == 'q':
                            ret[i][j] += 1
            else:
                ret[i][j] = 0
    
    return ret

def PairRelationshipOperator(img):
    # 0: background
    # 1: p
    # 2: q
    def func_h(a, m):
        if a == m:
            return 1
        else:
            return 0
    h, w = img.shape
    ret = np.zeros((h, w), np.uint8)
    for i in range(h):
        for j in range(w):
            if img[i][j] > 0:
                x1 = 0 if j + 1 == w else img[i][j+1]
                x2 = 0 if i - 1 < 0 else img[i-1][j]
                x3 = 0 if j - 1 < 0 else img[i][j-1]
                x4 = 0 if i + 1 == h else img[i+1][j]
                if (func_h(x1, 1) + func_h(x2, 1) + func_h(x3, 1) + func_h(x4, 1) > 0) and img[i][j] == 2:
                    ret[i][j] = 1
                else:
                    ret[i][j] = 2
    return ret

def MarkInteriorBorder(img):
    # 0: background
    # 1: interior
    # 2: border
    def func_h(c, d):
        if c == d: return c 
        else: return 2

    def func_f(c):
        if c == 2: return 2
        else: return 1

    h, w = img.shape
    ret = np.zeros((h, w), np.uint8)
    for i in range(h):
        for j in range(w):
            if img[i][j] > 0:
                x1 = 0 if j + 1 == w else 0 if img[i][j+1] == 0 else 1
                x2 = 0 if i - 1 < 0 else 0 if img[i-1][j] == 0 else 1
                x3 = 0 if j - 1 < 0 else 0 if img[i][j-1] == 0 else 1
                x4 = 0 if i + 1 == h else 0 if img[i+1][j] == 0 else 1
            
                a1 = func_h(1, x1)
                a2 = func_h(a1, x2)
                a3 = func_h(a2, x3)
                a4 = func_h(a3, x4)
                ret[i][j] = func_f(a4)
    
    return ret

def ThinningOperator(img):
    img = BinarizeImageAt128(img)
    img = resizeWith8x8(img)

    img_mem = CopyImg(img)
    buf = []
    while True:
        img_thin = CopyImg(img_mem)
        img_mib = MarkInteriorBorder(img_thin)
        img_pro = PairRelationshipOperator(img_mib)
        img_yok = YokoiConnectivityNumber(img_thin)

        for i in range(img_pro.shape[0]):
            for j in range(img_pro.shape[1]):
                if img_yok[i][j] == 1 and img_pro[i][j] == 1:
                    img_thin[i][j] = 0
        
        buf.append(img_thin)

        if (img_thin == img_mem).all():
            buf.pop()
            imageio.mimsave("./HW7/ThinningOperator.gif", buf, "GIF", duration=0.5)
            return img_thin
        else:
            img_mem = CopyImg(img_thin)

if __name__ == "__main__":
    img = cv2.imread("./HW7/lena.bmp", cv2.IMREAD_GRAYSCALE)
    img = ThinningOperator(img)
    cv2.imwrite("./HW7/ThinningOperator.bmp", img)