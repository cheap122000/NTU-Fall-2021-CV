import cv2
import numpy as np
import math

BLACK = 0
WHITE = 255

class EdgeDetector():
    def __init__(self, img, mask):
        self.img = img
        self.h, self.w = self.img.shape
        self.ret = np.zeros((img.shape), np.uint8)
        self.mask = mask

    def zero_crossing_detector(self, threshold, alpha):
        for i in range(self.h):
            for j in range(self.w):
                self.ret[i][j] = 1 if self.conv(i, j, alpha) >= threshold else -1 if self.conv(i, j, alpha) <= -threshold else 0
                # self.ret[i][j] = BLACK if self.conv(i, j, alpha) >= threshold else WHITE

        kernel_size = len(self.mask)
        half = kernel_size // 2

        for x in range(self.h):
            for y in range(self.w):
                if self.ret[i][j] == 1:
                    for i in range(-half + 1, half + 1):
                        for j in range(-half + 1, half + 1):
                            if x + i >= 0 and x + i <= kernel_size

        # x = position[0]
        # y = position[1]
        # rawSize = len(labels)
        # half = sizes[0]/2

        # if labels[x][y] == 1:
        #     for row in xrange(-half,  half+1):
        #         for col in xrange(-half, half+1):
        #             if x+col >= 0 and x+col <= rawSize-1 and y+row >= 0 and y+row <=rawSize-1:
        #                 # print x+col, y+row
        #                 if labels[x+col][y+row] == -1:
        #                     return 0

        # for ai in range(ra):
        #     for aj in range(ca):
        #         # check the sudden change of pixel magnitude for edge detection
        #         edge = 255
        #         if img_in[ai, aj] == 1:
        #             for ki in range(-rk // 2 + 1, rk // 2 + 1):
        #                 for kj in range(-ck // 2 + 1, ck // 2 + 1):
        #                     if  ai + ki >= 0 and ai + ki < ra and aj + kj >= 0 and aj + kj < ca:
        #                         if img_in[ai + ki, aj + kj] == -1:
        #                             edge = 0
        #         res[ai, aj] = edge

        # return WHITE 
        
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

    def conv(self, i, j, alpha):
        temp_matrix = self.get_extend_matrix(i, j)
        # temp = 0
        # temp += (temp_matrix * self.mask).sum()
        temp = (temp_matrix * self.mask).sum()
        temp *= alpha
        return temp

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
    cv2.imwrite("./HW10/laplacian1.bmp", EdgeDetector(img, mask1).zero_crossing_detector(15, 1))
    cv2.imwrite("./HW10/laplacian2.bmp", EdgeDetector(img, mask2).zero_crossing_detector(15, 1/3))
    cv2.imwrite("./HW10/laplacian3.bmp", EdgeDetector(img, mask3).zero_crossing_detector(20, 1/3))