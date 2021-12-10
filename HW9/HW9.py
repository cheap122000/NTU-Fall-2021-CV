import cv2
import numpy as np
import math

BLACK = 0
WHITE = 255

class EdgeDetector():
    def __init__(self, img):
        self.img = img
        self.h, self.w = self.img.shape
        self.ret = np.zeros((img.shape), np.uint8)

    def get_image(self, threshold):
        for i in range(self.h):
            for j in range(self.w):
                self.ret[i][j] = BLACK if self.get_magnitude(i, j) >= threshold else WHITE
        return self.ret

    def get_image_max(self, threshold):
        for i in range(self.h):
            for j in range(self.w):
                self.ret[i][j] = BLACK if self.get_max_magnitude(i, j) >= threshold else WHITE
        return self.ret

    def get_extend_matrix(self, i, j):
        if self.masks[0].shape[0] == 2:
            ret = np.array([
                [self.img[i][j], self.img[i][j+1] if j+1 < self.w else self.img[i][j]], 
                [self.img[i+1][j] if i+1 < self.h else self.img[i][j], self.img[i+1][j+1] if i+1 < self.h and j+1 < self.w else self.img[i][j] if i+1 >= self.h and j+1 >= self.w else self.img[i+1][j] if i+1 < self.h and j+1 >= self.w else self.img[i][j+1]]
            ])
            return ret
        else:
            ret = np.zeros((self.masks[0].shape), np.uint8)
            half = int(self.masks[0].shape[0] / 2)
            for h in range(-half, half+1):
                for w in range(-half, half+1):
                    if (i+h < 0 or i+h >= self.h) and (0 <= j+w < self.w):
                        ret[h+half][w+half] = img[i][j+w]
                    elif (j+w < 0 or j+w >= self.w) and (0 <= i+h < self.h):
                        ret[h+half][w+half] = img[i+h][j]
                    elif (0 <= j+w < self.w) and (0 <= i+h < self.h):
                        ret[h+half][w+half] = img[i+h][j+w]
                    else:
                        ret[h+half][w+half] = img[i][j]
            return ret


    
    def get_magnitude(self, i ,j):
        temp_matrix = self.get_extend_matrix(i, j)
        temp = 0
        for item in self.masks:
            temp += ((temp_matrix * item).sum()) ** 2
        return int(math.sqrt(temp))

    def get_max_magnitude(self, i ,j):
        temp_matrix = self.get_extend_matrix(i, j)
        temp = -np.inf
        for item in self.masks:
            temp = max(temp, (temp_matrix * item).sum())
        return temp

class RobertOperator(EdgeDetector):
    def __init__(self, img):
        super().__init__(img)
        self.masks = np.array([
            [
                [-1, 0], 
                [0, 1]
            ],
            [
                [0, -1], 
                [1, 0]
            ]
        ])

class PrewittEdgeDetector(EdgeDetector):
    def __init__(self, img):
        super().__init__(img)
        self.masks = np.array([
            [
                [-1, -1, -1], 
                [0, 0, 0],
                [1, 1, 1]
            ],
            [
                [-1, 0, 1], 
                [-1, 0, 1], 
                [-1, 0, 1]
            ]
        ])

class SobelEdgeDetector(EdgeDetector):
    def __init__(self, img):
        super().__init__(img)
        self.masks = np.array([
            [
                [-1, -2, -1], 
                [0, 0, 0],
                [1, 2, 1]
            ],
            [
                [-1, 0, 1], 
                [-2, 0, 2], 
                [-1, 0, 1]
            ]
        ])

class FreiAndChenGradientOperator(EdgeDetector):
    def __init__(self, img):
        super().__init__(img)
        self.masks = np.array([
            [
                [-1, -1*np.sqrt(2), -1], 
                [0, 0, 0],
                [1, np.sqrt(2), 1]
            ],
            [
                [-1, 0, 1], 
                [-1*np.sqrt(2), 0, np.sqrt(2)], 
                [-1, 0, 1]
            ]
        ])

class KirschCompassOperator(EdgeDetector):
    def __init__(self, img):
        super().__init__(img)
        self.masks = np.array([
            [
                [-3, -3, 5], 
                [-3, 0, 5],
                [-3, -3, 5]
            ],
            [
                [-3, 5, 5], 
                [-3, 0, 5],
                [-3, -3, -3]
            ],
            [
                [5, 5, 5], 
                [-3, 0, -3],
                [-3, -3, -3]
            ],
            [
                [5, 5, -3], 
                [5, 0, -3],
                [-3, -3, -3]
            ],
            [
                [5, -3, -3], 
                [5, 0, -3],
                [5, -3, -3]
            ],
            [
                [-3, -3, -3], 
                [5, 0, -3],
                [5, 5, -3]
            ],
            [
                [-3, -3, -3], 
                [-3, 0, -3],
                [5, 5, 5]
            ],
            [
                [-3, -3, -3], 
                [-3, 0, 5],
                [-3, 5, 5]
            ]
        ])

class RobinsonCompassOperator(EdgeDetector):
    def __init__(self, img):
        super().__init__(img)
        self.masks = np.array([
            [
                [-1, 0, 1], 
                [-2, 0, 2],
                [-1, 0, 1]
            ],
            [
                [0, 1, 2], 
                [-1, 0, 1],
                [-2, -1, 0]
            ],
            [
                [1, 2, 1], 
                [0, 0, 0],
                [-1, -2, -1]
            ],
            [
                [2, 1, 0], 
                [1, 0, -1],
                [0, -1, -2]
            ],
            [
                [1, 0, -1], 
                [2, 0, -2],
                [1, 0, -1]
            ],
            [
                [0, -1, -2], 
                [1, 0, -1],
                [2, 1, 0]
            ],
            [
                [-1, -2, -1], 
                [0, 0, 0],
                [1, 2, 1]
            ],
            [
                [-2, -1, 0], 
                [-1, 0, 1],
                [0, 1, 2]
            ]
        ])

class NevatiaBabu5x5Operator(EdgeDetector):
    def __init__(self, img):
        super().__init__(img)
        self.masks = np.array([
            [
                [100, 100, 100, 100, 100], 
                [100, 100, 100, 100, 100], 
                [0, 0, 0, 0, 0],
                [-100, -100, -100, -100, -100], 
                [-100, -100, -100, -100, -100]
            ],
            [
                [100, 100, 100, 100, 100], 
                [100, 100, 100, 78, -32],
                [100, 92, 0, -92, -100],
                [32, -78, -100, -100, -100],
                [-100, -100, -100, -100, -100]
            ],
            [
                [100, 100, 100, 32, -100], 
                [100, 100, 92, -78, -100],
                [100, 100, 0, -100, -100],
                [100, 78, -92, -100, -100],
                [100, -32, -100, -100, -100]
            ],
            [
                [-100, -100, 0, 100, 100], 
                [-100, -100, 0, 100, 100],
                [-100, -100, 0, 100, 100],
                [-100, -100, 0, 100, 100],
                [-100, -100, 0, 100, 100]
            ],
            [
                [-100, 32, 100, 100, 100], 
                [-100, -78, 92, 100, 100],
                [-100, -100, 0, 100, 100],
                [-100, -100, -92, 78, 100],
                [-100, -100, -100, -32, 100]
            ],
            [
                [100, 100, 100, 100, 100], 
                [-32, 78, 100, 100, 100],
                [-100, -92, 0, 92, 100],
                [-100, -100, -100, -78, 32],
                [-100, -100, -100, -100, -100]
            ],
        ])

if __name__ == "__main__":
    img = cv2.imread("./HW9/lena.bmp", cv2.IMREAD_GRAYSCALE)
    cv2.imwrite("./HW9/RobertsOperator.bmp", RobertOperator(img).get_image(12))
    cv2.imwrite("./HW9/PrewittsEdgeDetector.bmp", PrewittEdgeDetector(img).get_image(24))
    cv2.imwrite("./HW9/SobelsEdgeDetector.bmp", SobelEdgeDetector(img).get_image(38))
    cv2.imwrite("./HW9/FreiAndChensGradientOperator.bmp", FreiAndChenGradientOperator(img).get_image(30))
    cv2.imwrite("./HW9/KirschCompassOperator.bmp", KirschCompassOperator(img).get_image_max(135))
    cv2.imwrite("./HW9/RobinsonCompassOperator.bmp", RobinsonCompassOperator(img).get_image_max(43))
    cv2.imwrite("./HW9/NevatiaBabu5x5Operator.bmp", NevatiaBabu5x5Operator(img).get_image_max(12500))