import cv2
import numpy as np

def dilationImg(img, kernel):
    h, w = img.shape
    ret = np.zeros((h, w), np.uint8)
    kernel_h, kernel_w = len(kernel), len(kernel[0])
    for i in range(0 - int(kernel_h/2), h - int(kernel_h/2)):
        for j in range(0 - int(kernel_w/2), w - int(kernel_w/2)):
            localMax = 0
            for m in range(kernel_h):
                for n in range(kernel_w):
                    if i+m < 0 or j+n < 0 or i+m >= h or j+n >= w: continue
                    if kernel[m][n] == 1:
                        localMax = max(localMax, img[i+m][j+n])
            
            ret[i+int(kernel_h/2)][j+int(kernel_w/2)] = localMax
    return ret

def erosionImg(img, kernel):
    h, w = img.shape
    ret = np.zeros((h, w), np.uint8)
    kernel_h, kernel_w = len(kernel), len(kernel[0])
    for i in range(0 - int(kernel_h/2), h - int(kernel_h/2)):
        for j in range(0 - int(kernel_w/2), w - int(kernel_w/2)):
            validate = True
            localMin = 255
            for m in range(kernel_h):
                for n in range(kernel_w):
                    if i+m < 0 or j+n < 0 or i+m >= h or j+n >= w: continue
                    if kernel[m][n] == 1 and img[i+m][j+n] > 0:
                        if img[i+m][j+n] > 0:
                            localMin = min(localMin, img[i+m][j+n])
                    elif kernel[m][n] == 1 and img[i+m][j+n] == 0:
                        validate = False
                        break
                if not validate:
                    break

            if validate:
                ret[i+int(kernel_h/2)][j+int(kernel_w/2)] = localMin
    return ret

def openingImg(img, kernel):
    return dilationImg(erosionImg(img, kernel), kernel)

def closingImg(img, kernel):
    return erosionImg(dilationImg(img, kernel), kernel)

if __name__ == "__main__":
    octogon = [
        [0,1,1,1,0],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [0,1,1,1,0]
    ]

    img = cv2.imread("./HW5/lena.bmp", cv2.IMREAD_GRAYSCALE)
    img = dilationImg(img, octogon)
    cv2.imwrite("./HW5/dilation.bmp", img)

    img = cv2.imread("./HW5/lena.bmp", cv2.IMREAD_GRAYSCALE)
    img = erosionImg(img, octogon)
    cv2.imwrite("./HW5/erosion.bmp", img)

    img = cv2.imread("./HW5/lena.bmp", cv2.IMREAD_GRAYSCALE)
    img = openingImg(img, octogon)
    cv2.imwrite("./HW5/opening.bmp", img)

    img = cv2.imread("./HW5/lena.bmp", cv2.IMREAD_GRAYSCALE)
    img = closingImg(img, octogon)
    cv2.imwrite("./HW5/closing.bmp", img)