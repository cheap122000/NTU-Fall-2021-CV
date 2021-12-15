import cv2
import numpy as np
import math

BLACK = 0
WHITE = 255

class EdgeDetector():
    def __init__(self, img, mask):
        self.img = img
        self.h, self.w = self.img.shape
        # self.ret = np.full(img.shape, 255, dtype=int) 
        self.ret = np.zeros((img.shape), dtype=int)
        self.mask = mask

    def zero_crossing_detector(self, threshold, alpha):
        labels = np.zeros((img.shape), dtype=int)
        for i in range(self.h):
            for j in range(self.w):
                labels[i][j] = self.conv(i, j, alpha, threshold)

        for i, l in enumerate(labels):
            if i < 20:
                print(l[0:20])
        
        raw_size = len(labels)
        half = len(self.mask) // 2

        for x in range(self.h):
            for y in range(self.w):
                edge = WHITE
                if labels[x][j] == 1:
                    for i in range(-half, half + 1):
                        for j in range(-half, half + 1):
                            if x + i >= 0 and x + i < raw_size and y + j >= 0 and y + j < raw_size:
                                # print("?")
                                if labels[x+i][y+j] == -1:
                                    edge = BLACK
                
                self.ret[x][y] = edge

        print(self.ret)
        print('='*10)
        
        return self.ret

    def get_extend_matrix(self, i, j):
        if self.mask.shape[0] == 2:
            ret = np.array([
                [self.img[i][j], self.img[i][j+1] if j+1 < self.w else self.img[i][j]], 
                [self.img[i+1][j] if i+1 < self.h else self.img[i][j], self.img[i+1][j+1] if i+1 < self.h and j+1 < self.w else self.img[i][j] if i+1 >= self.h and j+1 >= self.w else self.img[i+1][j] if i+1 < self.h and j+1 >= self.w else self.img[i][j+1]]
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
        temp = 0
        # temp += (temp_matrix * self.mask).sum()
        # temp = (temp_matrix * self.mask).sum() * alpha
        # temp *= alpha
        for i in range(len(temp_matrix)):
            for j in range(len(temp_matrix)):
                temp += temp_matrix[i][j] * self.mask[len(temp_matrix)-i-1][len(temp_matrix)-j-1]
        temp *= alpha
        # print(temp)
        
        return 1 if temp >= threshold else -1 if temp <= -threshold else 0

if __name__ == "__main__":
    img = cv2.imread("./HW10/lena.bmp", cv2.IMREAD_GRAYSCALE)

    mask1 = np.array(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        )
    mask2 = np.array(
            [[1, 1, 1], [1, -8, 1], [1, 1, 1]]
        )
    mask3 = np.array(
            [[2, -1, 2], [-1, -4, -1], [2, -1, 2]]
        )
    cv2.imwrite("./HW10/laplacian1.bmp", EdgeDetector(img, mask1).zero_crossing_detector(15, 1.0))
    # cv2.imwrite("./HW10/laplacian2.bmp", EdgeDetector(img, mask2).zero_crossing_detector(15, 1/3))
    # cv2.imwrite("./HW10/laplacian3.bmp", EdgeDetector(img, mask3).zero_crossing_detector(20, 1/3))