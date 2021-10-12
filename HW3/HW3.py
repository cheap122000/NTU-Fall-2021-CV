import cv2
import numpy as np
import matplotlib.pyplot as plt

def DrawHistogram(img_path, save_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    x = [i for i in range(256)]
    y = np.zeros(256, dtype=int)
    for i in range(h):
        for j in range(w):
            y[img[i][j]] += 1
    plt.figure().clear()
    plt.bar(x, y, width=1)
    plt.savefig(save_path)

def OriginHistogram(img):
    cv2.imwrite("./HW3/Ori_image.bmp", img)
    DrawHistogram("./HW3/Ori_image.bmp", "./HW3/Ori_hist.jpg")

def IntensityDevidedBy3(img):
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            img[i][j] = int(img[i][j] / 3)
    
    cv2.imwrite("./HW3/IntensityDevidedBy3.bmp", img)
    DrawHistogram("./HW3/IntensityDevidedBy3.bmp", "./HW3/IntensityDevidedBy3_hist.jpg")

def ApplyHistogramEqualization(img):
    statistics = [0 for _ in range(256)]
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            statistics[img[i][j]] += 1
    statistics = [sum(statistics[:i+1]) if statistics[i] else 0 for i in range(256)]

    for i in range(h):
        for j in range(w):
            img[i][j] = int(((statistics[img[i][j]] - 1) / (max(statistics)-min(statistics))) * 255)

    cv2.imwrite("./HW3/ApplyHistogramEqualization.bmp", img)
    DrawHistogram("./HW3/ApplyHistogramEqualization.bmp", "./HW3/ApplyHistogramEqualization_hist.jpg")

if __name__ == "__main__":
    img = cv2.imread("./HW3/lena.bmp", cv2.IMREAD_GRAYSCALE)
    OriginHistogram(img)

    img = cv2.imread("./HW3/lena.bmp", cv2.IMREAD_GRAYSCALE)
    IntensityDevidedBy3(img)

    img = cv2.imread("./HW3/IntensityDevidedBy3.bmp", cv2.IMREAD_GRAYSCALE)
    ApplyHistogramEqualization(img)