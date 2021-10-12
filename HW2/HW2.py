import cv2
import numpy as np
import matplotlib.pyplot as plt

def BinarizeImage(img):
    h, w = img.shape
    ret = np.zeros((h, w), np.uint8)
    for i in range(h):
        for j in range(w):
            if img[i][j] >= 128:
                ret[i][j] = 255
            else:
                ret[i][j] = 0
    cv2.imwrite("./HW2/binaryImage.bmp", ret)

def DrawHistogram(img):
    h, w = img.shape
    x = [i for i in range(256)]
    y = np.zeros(256, dtype=int)
    for i in range(h):
        for j in range(w):
            y[img[i][j]] += 1
    plt.bar(x, y, width=1)
    plt.savefig("./HW2/histograph.jpg")

def ConnectedComponents(img):
    h, w = img.shape
    ret = np.empty((h, w), dtype=int)
    ret.fill(0)
    label = []
    storage = {}

    for i in range(h):
        for j in range(w):
            if img[i][j] == 0:
                ret[i][j] == 0
            else:
                if j == 0:
                    if i == 0:
                        if img[i][j] == 0:
                            ret[i][j] = 0
                        else:
                            label.append(len(label))
                            storage[len(label)] = []
                            ret[i][j] = len(label)
                            storage[ret[i][j]].append((i, j))
                    else:
                        label.append(len(label))
                        storage[len(label)] = []
                        ret[i][j] = len(label)
                        storage[ret[i][j]].append((i, j))
                else:
                    if img[i][j] == img[i][j-1]:
                        ret[i][j] = ret[i][j-1]
                        storage[ret[i][j]].append((i, j))
                    else:
                        label.append(len(label))
                        storage[len(label)] = []
                        ret[i][j] = len(label)
                        storage[ret[i][j]].append((i, j))
        if i > 0:
            for j in range(w):
                if (ret[i-1][j] != 0 and ret[i][j] != 0) and (ret[i-1][j] != ret[i][j]):
                    if ret[i-1][j] <= ret[i][j]:
                        temp = ret[i][j]
                        for (a, b) in storage[ret[i][j]]:
                            storage[ret[i-1][j]].append((a, b))
                            ret[a][b] = ret[i-1][j]
                        storage[temp] = []
                    else:
                        temp = ret[i-1][j]
                        for (a, b) in storage[ret[i-1][j]]:
                            storage[ret[i][j]].append((a, b))
                            ret[a][b] = ret[i][j]
                        storage[temp] = []
                
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for k, item in storage.items():
        maxH, maxW, minH, minW = -1, -1, -1, -1
        points = []
        for (hh, ww) in item:
            points.append((ww, hh))
            maxH = hh if maxH == -1 or hh > maxH else maxH
            minH = hh if minH == -1 or hh < minH else minH
            maxW = ww if maxW == -1 or ww > maxW else maxW
            minW = ww if minW == -1 or ww < minW else minW
        area = (maxH-minH+1) * (maxW-minW+1)
        if len(item) > 500:
            x, y = get_gravity_point(points)
            print(x, y)
            cv2.rectangle(img, (minW, minH), (maxW, maxH), (255,0,0), 3, cv2.LINE_AA)
            # cv2.circle(img, (x,y), 2, (0,0,255), 8)
            cv2.rectangle(img, (x-10,y), (x+10,y), (0,0,255), 1, cv2.LINE_AA)
            cv2.rectangle(img, (x,y-10), (x,y+10), (0,0,255), 1, cv2.LINE_AA)
            
    
    cv2.imwrite("./HW2/ConnectedComponents.bmp", img)

def get_gravity_point(points):
    if len(points) <= 2:
        return list()

    area = float(len(points))
    x, y = 0.0, 0.0
    for i in range(len(points)):
        x += points[i][0]
        y += points[i][1]

    x = x / area
    y = y / area

    return int(x), int(y)

if __name__ == "__main__":
    img = cv2.imread("./HW2/lena.bmp", cv2.IMREAD_GRAYSCALE)
    BinarizeImage(img)

    img = cv2.imread("./HW2/lena.bmp", cv2.IMREAD_GRAYSCALE)
    DrawHistogram(img)

    img = cv2.imread("./HW2/binaryImage.bmp", cv2.IMREAD_GRAYSCALE)
    ConnectedComponents(img)