import cv2
import numpy as np
import imutils

def UpsideDown(img):
    height, weight, channel = img.shape
    ret = np.zeros((height, weight, channel), np.uint8)
    for h in range(height):
        for w in range(weight):
            ret[h][w] = img[height-h-1][w]
    cv2.imwrite("./HW1/UpsideDown.bmp", ret)

def RightSideLeft(img):
    height, weight, channel = img.shape
    ret = np.zeros((height, weight, channel), np.uint8)
    for h in range(height):
        for w in range(weight):
            ret[h][w] = img[h][weight-w-1]
    cv2.imwrite("./HW1/RightSideLeft.bmp", ret)

def DiagonallyFlip(img):
    height, weight, channel = img.shape
    ret = np.zeros((height, weight, channel), np.uint8)
    for h in range(height):
        for w in range(weight):
            if h != w:
                ret[h][w] = img[w][h]
            else:
                ret[h][w] = img[h][w]
    cv2.imwrite("./HW1/DiagonallyFlip.bmp", ret)

def Rotate45Clockwise(img):
    ret = imutils.rotate_bound(img, 45)
    cv2.imwrite("./HW1/Rotate45Clockwise.bmp", ret)

def ShringToHalf(img):
    height, weight, channel = img.shape
    ret = cv2.resize(img, (int(height/2), int(weight/2)), cv2.INTER_AREA)
    cv2.imwrite("./HW1/ShringToHalf.bmp", ret)

def BinarizeAt128(img):
    _, ret = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    cv2.imwrite("./HW1/BinarizeAt128.bmp", ret)

if __name__ == "__main__":
    img = cv2.imread("./HW1/lena.bmp")
    UpsideDown(img)
    RightSideLeft(img)
    DiagonallyFlip(img)
    Rotate45Clockwise(img)
    ShringToHalf(img)
    BinarizeAt128(img)
    temp = [f"{i:02}" for i in range(0,30)]
    from datetime import datetime, timedelta
    print((datetime.now()-timedelta(hours=1)).strftime('%H'))