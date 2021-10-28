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

def isOverlap(matrix1, kernel):
    for x in range(len(matrix1)):
        for y in range(len(matrix1[0])):
            if kernel[x][y] != 0:
                if matrix1[x][y] == 0:
                    return False
    return True

def dilationImg(img, origin, kernel):
    img = BinarizeImageAt128(img)
    h, w = img.shape
    ret = np.zeros((h, w), np.uint8)
    kernel_h, kernel_w = len(kernel), len(kernel[0])
    for i in range(h - kernel_h):
        for j in range(w - kernel_w):
            if img[i+origin[0]][j+origin[1]] != 0:
                for m in range(kernel_h):
                    for n in range(kernel_w):
                        if kernel[m][n] == 1:
                            ret[i+m][j+n] = 255
    return ret
    # cv2.imwrite("./HW4/dilation.bmp", ret)

def CompletementImg(img):
    h, w = img.shape
    ret = np.zeros((h, w), np.uint8)
    for i in range(h):
        for j in range(w):
            ret[i][j] = 255 if img[i][j] == 0 else 0

    return ret

def Intersection(img1, img2):
    h, w = img1.shape
    ret = np.zeros((h, w), np.uint8)
    for i in range(h):
        for j in range(w):
            if img1[i][j] == 255 and img2[i][j] == 255:
                ret[i][j] = 255

    return ret
            
def erosionImg(img, origin, kernel):
    img = BinarizeImageAt128(img)
    h, w = img.shape
    ret = np.zeros((h, w), np.uint8)
    temp = []
    kernel_h, kernel_w = len(kernel), len(kernel[0])
    for i in range(h - kernel_h):
        for j in range(w - kernel_w):
            temp = []
            for k in range(kernel_h):
                temp.append([img[i+k][j+m] for m in range(kernel_w)])
                if isOverlap(temp, kernel):
                    ret[i+origin[1]][j+origin[0]] = 255
                else:
                    ret[i+origin[1]][j+origin[0]] = 0
    return ret
    # cv2.imwrite("./HW4/erosion.bmp", ret)

def HitAndMiss(img, kernel):
    img = BinarizeImageAt128(img)
    img1 = erosionImg(img, (1,0), kernel)
    img2 = erosionImg(CompletementImg(img), (0,1), kernel)
    ret = Intersection(img1, img2)
    return ret

def OpeningImg(img, origin, kernel):
    return dilationImg(erosionImg(img, origin, kernel), origin, kernel)

def ClosingImg(img, origin, kernel):
    return erosionImg(dilationImg(img, origin, kernel), origin, kernel)


if __name__ == "__main__":
    octogon = [
        [0,1,1,1,0],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [0,1,1,1,0]
    ]

    Lpattern = [
        [1,1],
        [0,1]
    ]

    img = cv2.imread("./HW4/lena.bmp", cv2.IMREAD_GRAYSCALE)
    img = dilationImg(img, (2,2), octogon)
    cv2.imwrite("./HW4/dilation.bmp", img)

    img = cv2.imread("./HW4/lena.bmp", cv2.IMREAD_GRAYSCALE)
    img = erosionImg(img, (2,2), octogon)
    cv2.imwrite("./HW4/erosion.bmp", img)

    img = cv2.imread("./HW4/lena.bmp", cv2.IMREAD_GRAYSCALE)
    img = HitAndMiss(img, Lpattern)
    cv2.imwrite("./HW4/hitandmiss.bmp", img)

    img = cv2.imread("./HW4/lena.bmp", cv2.IMREAD_GRAYSCALE)
    img = OpeningImg(img, (2,2), octogon)
    cv2.imwrite("./HW4/opening.bmp", img)

    img = cv2.imread("./HW4/lena.bmp", cv2.IMREAD_GRAYSCALE)
    img = ClosingImg(img, (2,2), octogon)
    cv2.imwrite("./HW4/closing.bmp", img)