import cv2
import numpy as np
import math

BLACK = 0
WHITE = 255

class Detector():
    def __init__(self, img, mask):
        self.img = img
        self.h, self.w = self.img.shape
        self.ret = np.full(img.shape, 255, dtype=int) 
        self.mask = mask

    def zero_crossing_detector(self, threshold, alpha):
        labels = np.zeros((img.shape), dtype=int)
        for i in range(self.h):
            for j in range(self.w):
                labels[i][j] = self.conv(i, j, alpha, threshold)
        
        half = 3 // 2

        for x in range(self.h):
            for y in range(self.w):
                edge = WHITE
                if labels[x][y] == 1:
                    for i in range(-half, half + 1):
                        for j in range(-half, half + 1):
                            if x + i >= 0 and x + i < self.h and y + j >= 0 and y + j < self.w:
                                if labels[x+i][y+j] == -1:
                                    edge = BLACK
                
                self.ret[x][y] = edge
        
        return self.ret

    def get_extend_matrix(self, i, j):
        if self.mask.shape[0] == 2:
            ret = np.array([
                [self.img[i][j], self.img[i][j+1] if j+1 < self.w \
                    else self.img[i][j]], [self.img[i+1][j] if i+1 < self.h \
                    else self.img[i][j], self.img[i+1][j+1] if i+1 < self.h and j+1 < self.w \
                    else self.img[i][j] if i+1 >= self.h and j+1 >= self.w \
                    else self.img[i+1][j] if i+1 < self.h and j+1 >= self.w \
                    else self.img[i][j+1]]
            ])
            return ret
        else:
            ret = np.zeros((self.mask.shape), np.uint8)
            half = int(self.mask.shape[0] / 2)
            for h in range(-half, half+1):
                for w in range(-half, half+1):
                    if (i+h < 0 or i+h >= self.h) and (0 <= j+w < self.w):
                        ret[h+half][w+half] = self.img[i][j+w]
                    elif (j+w < 0 or j+w >= self.w) and (0 <= i+h < self.h):
                        ret[h+half][w+half] = self.img[i+h][j]
                    elif (0 <= j+w < self.w) and (0 <= i+h < self.h):
                        ret[h+half][w+half] = self.img[i+h][j+w]
                    else:
                        ret[h+half][w+half] = self.img[i][j]
            return ret

    def conv(self, i, j, alpha, threshold):
        temp_matrix = self.get_extend_matrix(i, j)
        # temp = (temp_matrix * self.mask).sum() * alpha
        temp = 0
        for i in range(len(temp_matrix)):
            for j in range(len(temp_matrix)):
                temp += temp_matrix[i][j] * self.mask[len(temp_matrix)-i-1][len(temp_matrix)-j-1]
        temp *= alpha
        
        return 1 if temp >= threshold else -1 if temp <= -threshold else 0

if __name__ == "__main__":
    img = cv2.imread("./HW10/lena.bmp", cv2.IMREAD_GRAYSCALE)

    mask = np.array(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        )
    cv2.imwrite("./HW10/laplacian1.bmp", Detector(img, mask).zero_crossing_detector(15, 1))
    
    mask = np.array(
            [[1, 1, 1], [1, -8, 1], [1, 1, 1]]
        )
    cv2.imwrite("./HW10/laplacian2.bmp", Detector(img, mask).zero_crossing_detector(15, 1/3))

    mask = np.array(
            [[2, -1, 2], [-1, -4, -1], [2, -1, 2]]
        )
    cv2.imwrite("./HW10/minimum_laplacian.bmp", Detector(img, mask).zero_crossing_detector(20, 1/3))

    mask = np.array([
            [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],
            [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
            [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
            [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
            [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
            [-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],
            [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
            [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
            [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
            [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
            [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0]
        ])
    cv2.imwrite("./HW10/laplacian_gaussian.bmp", Detector(img, mask).zero_crossing_detector(3000, 1))

    mask = np.array([
            [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
            [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
            [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
            [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
            [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
            [-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8],
            [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
            [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
            [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
            [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
            [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
        ])
    cv2.imwrite("./HW10/difference_gaussian.bmp", Detector(img, mask).zero_crossing_detector(1, 1))