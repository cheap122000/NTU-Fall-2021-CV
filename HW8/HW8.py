import cv2
from PIL import Image
import random
import numpy as np
import math

def getMedium(lst):
    lst = sorted(lst)
    if len(lst) % 2 != 0:
        return lst[int(len(lst)/2)]
    else:
        return (lst[int(len(lst)/2)-1] + lst[int(len(lst)/2)]) / 2

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

def GetGaussianNoise_Image(original_Image, amplitude):
    gaussianNoise_Image = original_Image.copy()
    for c in range(original_Image.size[0]):
        for r in range(original_Image.size[1]):
            noisePixel = int(original_Image.getpixel((c, r)) + amplitude * random.gauss(0, 1))
            noisePixel = 255 if noisePixel > 255 else noisePixel
            gaussianNoise_Image.putpixel((c, r), noisePixel)
    return gaussianNoise_Image

def GetSaultAndPepper_Image(original_Image, threshold):
    SaltAndPepper_Image = original_Image.copy()
    for c in range(original_Image.size[0]):
        for r in range(original_Image.size[1]):
            random_Value = random.uniform(0, 1)
            if random_Value <= threshold:
                SaltAndPepper_Image.putpixel((c, r), 0)
            elif random_Value >= 1 - threshold:
                SaltAndPepper_Image.putpixel((c, r), 255)
            else:
                SaltAndPepper_Image.putpixel((c, r), original_Image.getpixel((c, r)))
    return SaltAndPepper_Image

def BoxFilter(img, filter_size):
    h, w = img.shape
    ret = np.zeros((h, w), np.uint8)
    offset = int(filter_size/2)
    for i in range(h):
        for j in range(w):
            temp = []
            for c in range(i-offset, i+offset+1):
                for r in range(j-offset, j+offset+1):
                    if c >= 0 and c < h and r >= 0 and r < w:
                        temp.append(img[c][r])
            ret[i][j] = int(sum(temp)/len(temp))
    return ret

def MedianFilter(img, filter_size):
    h, w = img.shape
    ret = np.zeros((h, w), np.uint8)
    offset = int(filter_size/2)
    for i in range(h):
        for j in range(w):
            temp = []
            for c in range(i-offset, i+offset+1):
                for r in range(j-offset, j+offset+1):
                    if c >= 0 and c < h and r >= 0 and r < w:
                        temp.append(int(img[c][r]))
            ret[i][j] = int(getMedium(temp))
    return ret

def SNR(originalImage, noiseImage):
    h, w = originalImage.shape
    size = h * w
    us = 0
    vs = 0 
    un = 0
    vn = 0

    for i in range(h):
        for j in range(w):
            us += originalImage[i][j]
    us /= size

    for i in range(h):
        for j in range(w):
            vs += math.pow(originalImage[i][j] - us, 2)
    vs /= size

    for i in range(h):
        for j in range(w):
            un += int(noiseImage[i][j]) - int(originalImage[i][j])
    un /= size

    for i in range(h):
        for j in range(w):
            vn += math.pow(int(noiseImage[i][j]) - int(originalImage[i][j]) - int(un), 2)
    vn /= size

    return 20 * math.log(math.sqrt(vs) / math.sqrt(vn), 10)

if __name__ == "__main__":
    octogon = [
        [0,1,1,1,0],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [0,1,1,1,0]
    ]

    img = Image.open("./HW8/lena.bmp")

    GaussianNoise_10 = GetGaussianNoise_Image(img, 10)
    GaussianNoise_10.save("./HW8/GaussianNoise_10.bmp")
    GaussianNoise_30 = GetGaussianNoise_Image(img, 30)
    GaussianNoise_30.save("./HW8/GaussianNoise_30.bmp")

    SaltAndPepper_010 = GetSaultAndPepper_Image(img, 0.1)
    SaltAndPepper_010.save("./HW8/SaltAndPepper_0.1.bmp")
    SaltAndPepper_005 = GetSaultAndPepper_Image(img, 0.05)
    SaltAndPepper_005.save("./HW8/SaltAndPepper_0.05.bmp")

    cv2.imwrite("./HW8/Box_3x3_GaussianNoise_10.bmp", BoxFilter(cv2.imread("./HW8/GaussianNoise_10.bmp", cv2.IMREAD_GRAYSCALE), 3))
    cv2.imwrite("./HW8/Box_5x5_GaussianNoise_10.bmp", BoxFilter(cv2.imread("./HW8/GaussianNoise_10.bmp", cv2.IMREAD_GRAYSCALE), 5))
    cv2.imwrite("./HW8/Box_3x3_GaussianNoise_30.bmp", BoxFilter(cv2.imread("./HW8/GaussianNoise_30.bmp", cv2.IMREAD_GRAYSCALE), 3))
    cv2.imwrite("./HW8/Box_5x5_GaussianNoise_30.bmp", BoxFilter(cv2.imread("./HW8/GaussianNoise_30.bmp", cv2.IMREAD_GRAYSCALE), 5))
    cv2.imwrite("./HW8/Box_3x3_SaltAndPepper_0.1.bmp", BoxFilter(cv2.imread("./HW8/SaltAndPepper_0.1.bmp", cv2.IMREAD_GRAYSCALE), 3))
    cv2.imwrite("./HW8/Box_5x5_SaltAndPepper_0.1.bmp", BoxFilter(cv2.imread("./HW8/SaltAndPepper_0.1.bmp", cv2.IMREAD_GRAYSCALE), 5))
    cv2.imwrite("./HW8/Box_3x3_SaltAndPepper_0.05.bmp", BoxFilter(cv2.imread("./HW8/SaltAndPepper_0.05.bmp", cv2.IMREAD_GRAYSCALE), 3))
    cv2.imwrite("./HW8/Box_5x5_SaltAndPepper_0.05.bmp", BoxFilter(cv2.imread("./HW8/SaltAndPepper_0.05.bmp", cv2.IMREAD_GRAYSCALE), 5))

    cv2.imwrite("./HW8/Median_3x3_GaussianNoise_10.bmp", MedianFilter(cv2.imread("./HW8/GaussianNoise_10.bmp", cv2.IMREAD_GRAYSCALE), 3))
    cv2.imwrite("./HW8/Median_5x5_GaussianNoise_10.bmp", MedianFilter(cv2.imread("./HW8/GaussianNoise_10.bmp", cv2.IMREAD_GRAYSCALE), 5))
    cv2.imwrite("./HW8/Median_3x3_GaussianNoise_30.bmp", MedianFilter(cv2.imread("./HW8/GaussianNoise_30.bmp", cv2.IMREAD_GRAYSCALE), 3))
    cv2.imwrite("./HW8/Median_5x5_GaussianNoise_30.bmp", MedianFilter(cv2.imread("./HW8/GaussianNoise_30.bmp", cv2.IMREAD_GRAYSCALE), 5))
    cv2.imwrite("./HW8/Median_3x3_SaltAndPepper_0.1.bmp", MedianFilter(cv2.imread("./HW8/SaltAndPepper_0.1.bmp", cv2.IMREAD_GRAYSCALE), 3))
    cv2.imwrite("./HW8/Median_5x5_SaltAndPepper_0.1.bmp", MedianFilter(cv2.imread("./HW8/SaltAndPepper_0.1.bmp", cv2.IMREAD_GRAYSCALE), 5))
    cv2.imwrite("./HW8/Median_3x3_SaltAndPepper_0.05.bmp", MedianFilter(cv2.imread("./HW8/SaltAndPepper_0.05.bmp", cv2.IMREAD_GRAYSCALE), 3))
    cv2.imwrite("./HW8/Median_5x5_SaltAndPepper_0.05.bmp", MedianFilter(cv2.imread("./HW8/SaltAndPepper_0.05.bmp", cv2.IMREAD_GRAYSCALE), 5))

    cv2.imwrite("./HW8/_OpenClose_GaussianNoise_10.bmp", closingImg(openingImg(cv2.imread("./HW8/GaussianNoise_10.bmp", cv2.IMREAD_GRAYSCALE), octogon), octogon))
    cv2.imwrite("./HW8/_CloseOpen_GaussianNoise_10.bmp", openingImg(closingImg(cv2.imread("./HW8/GaussianNoise_10.bmp", cv2.IMREAD_GRAYSCALE), octogon), octogon))
    cv2.imwrite("./HW8/_OpenClose_GaussianNoise_30.bmp", closingImg(openingImg(cv2.imread("./HW8/GaussianNoise_30.bmp", cv2.IMREAD_GRAYSCALE), octogon), octogon))
    cv2.imwrite("./HW8/_CloseOpen_GaussianNoise_30.bmp", openingImg(closingImg(cv2.imread("./HW8/GaussianNoise_30.bmp", cv2.IMREAD_GRAYSCALE), octogon), octogon))
    cv2.imwrite("./HW8/_OpenClose_SaltAndPepper_0.1.bmp", closingImg(openingImg(cv2.imread("./HW8/SaltAndPepper_0.1.bmp", cv2.IMREAD_GRAYSCALE), octogon), octogon))
    cv2.imwrite("./HW8/_CloseOpen_SaltAndPepper_0.1.bmp", openingImg(closingImg(cv2.imread("./HW8/SaltAndPepper_0.1.bmp", cv2.IMREAD_GRAYSCALE), octogon), octogon))
    cv2.imwrite("./HW8/_OpenClose_SaltAndPepper_0.05.bmp", closingImg(openingImg(cv2.imread("./HW8/SaltAndPepper_0.05.bmp", cv2.IMREAD_GRAYSCALE), octogon), octogon))
    cv2.imwrite("./HW8/_CloseOpen_SaltAndPepper_0.05.bmp", openingImg(closingImg(cv2.imread("./HW8/SaltAndPepper_0.05.bmp", cv2.IMREAD_GRAYSCALE), octogon), octogon))


    img = cv2.imread("./HW8/lena.bmp", cv2.IMREAD_GRAYSCALE)
    with open("./HW8/results.txt", "w", encoding="utf8") as f:
        f.write(f"GaussianNoise_10:   {SNR(img, cv2.imread('./HW8/GaussianNoise_10.bmp', cv2.IMREAD_GRAYSCALE)): f}\n")
        f.write(f"GaussianNoise_30:   {SNR(img, cv2.imread('./HW8/GaussianNoise_30.bmp', cv2.IMREAD_GRAYSCALE)): f}\n")
        f.write(f"SaltAndPepper_0.05: {SNR(img, cv2.imread('./HW8/SaltAndPepper_0.05.bmp', cv2.IMREAD_GRAYSCALE)): f}\n")
        f.write(f"SaltAndPepper_0.1:  {SNR(img, cv2.imread('./HW8/SaltAndPepper_0.1.bmp', cv2.IMREAD_GRAYSCALE)): f}\n")

        f.write(f"Box_3x3_GaussianNoise_10:   {SNR(img, cv2.imread('./HW8/Box_3x3_GaussianNoise_10.bmp', cv2.IMREAD_GRAYSCALE)): f}\n")
        f.write(f"Box_5x5_GaussianNoise_10:   {SNR(img, cv2.imread('./HW8/Box_5x5_GaussianNoise_10.bmp', cv2.IMREAD_GRAYSCALE)): f}\n")
        f.write(f"Box_3x3_GaussianNoise_30:   {SNR(img, cv2.imread('./HW8/Box_3x3_GaussianNoise_30.bmp', cv2.IMREAD_GRAYSCALE)): f}\n")
        f.write(f"Box_5x5_GaussianNoise_30:   {SNR(img, cv2.imread('./HW8/Box_5x5_GaussianNoise_30.bmp', cv2.IMREAD_GRAYSCALE)): f}\n")
        f.write(f"Box_3x3_SaltAndPepper_0.1:  {SNR(img, cv2.imread('./HW8/Box_3x3_SaltAndPepper_0.1.bmp', cv2.IMREAD_GRAYSCALE)): f}\n")
        f.write(f"Box_5x5_SaltAndPepper_0.1:  {SNR(img, cv2.imread('./HW8/Box_5x5_SaltAndPepper_0.1.bmp', cv2.IMREAD_GRAYSCALE)): f}\n")
        f.write(f"Box_3x3_SaltAndPepper_0.05: {SNR(img, cv2.imread('./HW8/Box_3x3_SaltAndPepper_0.05.bmp', cv2.IMREAD_GRAYSCALE)): f}\n")
        f.write(f"Box_5x5_SaltAndPepper_0.05: {SNR(img, cv2.imread('./HW8/Box_5x5_SaltAndPepper_0.05.bmp', cv2.IMREAD_GRAYSCALE)): f}\n")

        f.write(f"Median_3x3_GaussianNoise_10:   {SNR(img, cv2.imread('./HW8/Median_3x3_GaussianNoise_10.bmp', cv2.IMREAD_GRAYSCALE)): f}\n")
        f.write(f"Median_5x5_GaussianNoise_10:   {SNR(img, cv2.imread('./HW8/Median_5x5_GaussianNoise_10.bmp', cv2.IMREAD_GRAYSCALE)): f}\n")
        f.write(f"Median_3x3_GaussianNoise_30:   {SNR(img, cv2.imread('./HW8/Median_3x3_GaussianNoise_30.bmp', cv2.IMREAD_GRAYSCALE)): f}\n")
        f.write(f"Median_5x5_GaussianNoise_30:   {SNR(img, cv2.imread('./HW8/Median_5x5_GaussianNoise_30.bmp', cv2.IMREAD_GRAYSCALE)): f}\n")
        f.write(f"Median_3x3_SaltAndPepper_0.1:  {SNR(img, cv2.imread('./HW8/Median_3x3_SaltAndPepper_0.1.bmp', cv2.IMREAD_GRAYSCALE)): f}\n")
        f.write(f"Median_5x5_SaltAndPepper_0.1:  {SNR(img, cv2.imread('./HW8/Median_5x5_SaltAndPepper_0.1.bmp', cv2.IMREAD_GRAYSCALE)): f}\n")
        f.write(f"Median_3x3_SaltAndPepper_0.05: {SNR(img, cv2.imread('./HW8/Median_3x3_SaltAndPepper_0.05.bmp', cv2.IMREAD_GRAYSCALE)): f}\n")
        f.write(f"Median_5x5_SaltAndPepper_0.05: {SNR(img, cv2.imread('./HW8/Median_5x5_SaltAndPepper_0.05.bmp', cv2.IMREAD_GRAYSCALE)): f}\n")

        f.write(f"OpenClose_GaussianNoise_10:   {SNR(img, cv2.imread('./HW8/_OpenClose_GaussianNoise_10.bmp', cv2.IMREAD_GRAYSCALE)): f}\n")
        f.write(f"CloseOpen_GaussianNoise_10:   {SNR(img, cv2.imread('./HW8/_CloseOpen_GaussianNoise_10.bmp', cv2.IMREAD_GRAYSCALE)): f}\n")
        f.write(f"OpenClose_GaussianNoise_30:   {SNR(img, cv2.imread('./HW8/_OpenClose_GaussianNoise_30.bmp', cv2.IMREAD_GRAYSCALE)): f}\n")
        f.write(f"CloseOpen_GaussianNoise_30:   {SNR(img, cv2.imread('./HW8/_CloseOpen_GaussianNoise_30.bmp', cv2.IMREAD_GRAYSCALE)): f}\n")
        f.write(f"OpenClose_SaltAndPepper_0.1:  {SNR(img, cv2.imread('./HW8/_OpenClose_SaltAndPepper_0.1.bmp', cv2.IMREAD_GRAYSCALE)): f}\n")
        f.write(f"CloseOpen_SaltAndPepper_0.1:  {SNR(img, cv2.imread('./HW8/_CloseOpen_SaltAndPepper_0.1.bmp', cv2.IMREAD_GRAYSCALE)): f}\n")
        f.write(f"OpenClose_SaltAndPepper_0.05: {SNR(img, cv2.imread('./HW8/_OpenClose_SaltAndPepper_0.05.bmp', cv2.IMREAD_GRAYSCALE)): f}\n")
        f.write(f"CloseOpen_SaltAndPepper_0.05: {SNR(img, cv2.imread('./HW8/_CloseOpen_SaltAndPepper_0.05.bmp', cv2.IMREAD_GRAYSCALE)): f}\n")