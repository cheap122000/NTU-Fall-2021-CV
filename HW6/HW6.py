import cv2
import numpy as np

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

def func_h(b, c, d, e):
    if b == c and (d != b or e != b): 
        return 'q'
    if b == c and (d == b and e == b): 
        return 'r'
    return 's'

def YokoConnectivityNumber(img):
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

                # print(x0, x1, x6, x2, [a1, a2, a3, a4])
                if [a1, a2, a3, a4] == ['r', 'r', 'r', 'r']:
                    ret[i][j] = 5
                else:
                    ret[i][j] = 0
                    for a in [a1, a2, a3, a4]:
                        if a == 'q':
                            ret[i][j] += 1
            else:
                ret[i][j] = 0
    
    with open('./HW6/result.txt', 'w', encoding='utf8') as f:
        for i, item in enumerate(ret):
            for elem in item:
                f.write(str(elem) if elem > 0 else " ")
            if i < 63:
                f.write('\n')

if __name__ == "__main__":
    img = cv2.imread("./HW6/lena.bmp", cv2.IMREAD_GRAYSCALE)
    img = BinarizeImageAt128(img)
    img = resizeWith8x8(img)
    cv2.imwrite('./HW6/resizeWith8x8.bmp', img)
    img = YokoConnectivityNumber(img)
