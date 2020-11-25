import cv2
import numpy as np
import random
import math

def GaussianNoise(img, amplitude):
    new_img = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img[i][j] = int(img[i][j] + amplitude * random.gauss(0, 1))
            if new_img[i][j] > 255:
                new_img[i][j] = 255
    return new_img

def SaltAndPepperNoise(img, threshold):
    new_img = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            random_variable = random.uniform(0, 1)
            if random_variable < threshold:
                new_img[i][j] = 0
            elif random_variable > 1 - threshold:
                new_img[i][j] = 255
            else:
                new_img[i][j] = img[i][j]
    return new_img

def BoxFilter(img, n):
    new_img = img.copy()
    half_n = int(n / 2)
    for i in range(half_n, img.shape[0]-half_n):
        for j in range(half_n, img.shape[1]-half_n):
            sum = 0
            for x in range(-half_n, half_n+1):
                for y in range(-half_n, half_n+1):
                    sum += img[i+x][j+y]
            new_img[i][j] = int(sum / (n * n))
    return new_img

def Median(array, n):
    for i in range(n-1):
        for j in range(n-i-1):
            if array[j] > array[j+1]:
                tmp = array[j]
                array[j] = array[j+1]
                array[j+1] = tmp
    return array[int(n / 2)]

def MedianFilter(img, n):
    new_img = img.copy()
    half_n = int(n / 2)
    for i in range(half_n, img.shape[0]-half_n):
        for j in range(half_n, img.shape[1]-half_n):
            tmp = [0] * (n * n)
            for x in range(-half_n, half_n+1):
                for y in range(-half_n, half_n+1):
                    tmp[(x+half_n)*n+(y+half_n)] = img[i+x][j+y]
            new_img[i][j] = Median(tmp, n*n)
    return new_img

#set kernel
ker = np.zeros([21, 2], dtype='int8')
#octogonal 3-5-5-5-3 kernel
ker[0] = [-2, -1]
ker[1] = [-2, 0]
ker[2] = [-2, 1]
ker[3] = [-1, -2]
ker[4] = [-1, -1]
ker[5] = [-1, 0]
ker[6] = [-1, 1]
ker[7] = [-1, 2]
ker[8] = [0, -2]
ker[9] = [0, -1]
ker[10] = [0, 0]
ker[11] = [0, 1]
ker[12] = [0, 2]
ker[13] = [1, -2]
ker[14] = [1, -1]
ker[15] = [1, 0]
ker[16] = [1, 1]
ker[17] = [1, 2]
ker[18] = [2, -1]
ker[19] = [2, 0]
ker[20] = [2, 1]

#gray scale dilation
def Dilation(img):
    new_img = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            max = 0
            for k in range(21):
                x, y = i - ker[k][0], j - ker[k][1]
                if (x >= 0 and x < img.shape[0] and y >= 0 and y < img.shape[1]):
                    if (img[x][y] > max):
                        max = img[x][y]
            new_img[i][j] = max
    return new_img

#gray scale erosion
def Erosion(img):
    new_img = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            min = 255
            for k in range(21):
                x, y = i + ker[k][0], j + ker[k][1]
                if (x >= 0 and x < img.shape[0] and y >= 0 and y < img.shape[1]):
                    if (img[x][y] < min):
                        min = img[x][y]
            new_img[i][j] = min
    return new_img

#gray scale opening
def Opening(img):
    new_img = img.copy()
    new_img = img
    return Dilation(Erosion(new_img))

#gray scale closing
def Closing(img):
    new_img = img.copy()
    new_img = img
    return Erosion(Dilation(new_img))

#Calculate SNR
def SNR(img, img_n):
    img = img.astype(np.float) / 255.0
    img_n = img_n.astype(np.float) / 255.0
    img_diff = img - img_n
    std_s = np.std(img)
    std_n = np.std(img_diff)
    return round(20 * math.log(std_s / std_n, 10), 3)

img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

#Generate additive white Gaussian noise
a_1 = GaussianNoise(img, 10)
cv2.imwrite('a-1.jpg', a_1)
print('a-1.jpg SNR = ', SNR(img, a_1))
a_2 = GaussianNoise(img, 30)
cv2.imwrite('a-2.jpg', a_2)
print('a-2.jpg SNR = ', SNR(img, a_2))

#Generate salt-and-pepper noise
b_1 = SaltAndPepperNoise(img, 0.1)
cv2.imwrite('b-1.jpg', b_1)
print('b-1.jpg SNR = ', SNR(img, b_1))

b_2 = SaltAndPepperNoise(img, 0.05)
cv2.imwrite('b-2.jpg', b_2)
print('b-2.jpg SNR = ', SNR(img, b_2))

#Use the 3x3, 5x5 box filter
box3_1 = BoxFilter(a_1, 3)
cv2.imwrite('box3-1.jpg', box3_1)
print('box3-1.jpg SNR = ', SNR(img, box3_1))

box3_2 = BoxFilter(a_2, 3)
cv2.imwrite('box3-2.jpg', box3_2)
print('box3-2.jpg SNR = ', SNR(img, box3_2))

box3_3 = BoxFilter(b_1, 3)
cv2.imwrite('box3-3.jpg', box3_3)
print('box3-3.jpg SNR = ', SNR(img, box3_3))

box3_4 = BoxFilter(b_2, 3)
cv2.imwrite('box3-4.jpg', box3_4)
print('box3-4.jpg SNR = ', SNR(img, box3_4))

box5_1 = BoxFilter(a_1, 5)
cv2.imwrite('box5-1.jpg', box5_1)
print('box5-1.jpg SNR = ', SNR(img, box5_1))

box5_2 = BoxFilter(a_2, 5)
cv2.imwrite('box5-2.jpg', box5_2)
print('box5-2.jpg SNR = ', SNR(img, box5_2))

box5_3 = BoxFilter(b_1, 5)
cv2.imwrite('box5-3.jpg', box5_3)
print('box5-3.jpg SNR = ', SNR(img, box5_3))

box5_4 = BoxFilter(b_2, 5)
cv2.imwrite('box5-4.jpg', box5_4)
print('box5-4.jpg SNR = ', SNR(img, box5_4))

#Use 3x3, 5x5 median filter
median3_1 = MedianFilter(a_1, 3)
cv2.imwrite('median3-1.jpg', median3_1)
print('median3-1.jpg SNR = ', SNR(img, median3_1))

median3_2 = MedianFilter(a_2, 3)
cv2.imwrite('median3-2.jpg', median3_2)
print('median3-2.jpg SNR = ', SNR(img, median3_2))

median3_3 = MedianFilter(b_1, 3)
cv2.imwrite('median3-3.jpg', median3_3)
print('median3-3.jpg SNR = ', SNR(img, median3_3))

median3_4 = MedianFilter(b_2, 3)
cv2.imwrite('median3-4.jpg', median3_4)
print('median3-4.jpg SNR = ', SNR(img, median3_4))

median5_1 = MedianFilter(a_1, 5)
cv2.imwrite('median5-1.jpg', median5_1)
print('median5-1.jpg SNR = ', SNR(img, median5_1))

median5_2 = MedianFilter(a_2, 5)
cv2.imwrite('median5-2.jpg', median5_2)
print('median5-2.jpg SNR = ', SNR(img, median5_2))

median5_3 = MedianFilter(b_1, 5)
cv2.imwrite('median5-3.jpg', median5_3)
print('median5-3.jpg SNR = ', SNR(img, median5_3))

median5_4 = MedianFilter(b_2, 5)
cv2.imwrite('median5-4.jpg', median5_4)
print('median5-4.jpg SNR = ', SNR(img, median5_4))

#Use both opening-then-closing and closing-then opening filter
#cv2.imwrite('oc-1.jpg', Closing(Opening(gaussian_noise_10)))
oc_1 = Closing(Opening(a_1))
cv2.imwrite('oc-1.jpg', oc_1)
print('oc-1.jpg SNR = ', SNR(img, oc_1))

oc_2 = Closing(Opening(a_2))
cv2.imwrite('oc-2.jpg', oc_2)
print('oc-2.jpg SNR = ', SNR(img, oc_2))

oc_3 = Closing(Opening(b_1))
cv2.imwrite('oc-3.jpg', oc_3)
print('oc-3.jpg SNR = ', SNR(img, oc_3))

oc_4 = Closing(Opening(b_2))
cv2.imwrite('oc-4.jpg', oc_4)
print('oc-4.jpg SNR = ', SNR(img, oc_4))

co_1 = Opening(Closing(a_1))
cv2.imwrite('co-1.jpg', co_1)
print('co-1.jpg SNR = ', SNR(img, co_1))

co_2 = Opening(Closing(a_2))
cv2.imwrite('co-2.jpg', co_2)
print('co-2.jpg SNR = ', SNR(img, co_2))

co_3 = Opening(Closing(b_1))
cv2.imwrite('co-3.jpg', co_3)
print('co-3.jpg SNR = ', SNR(img, co_3))

co_4 = Opening(Closing(b_2))
cv2.imwrite('co-4.jpg', co_4)
print('co-4.jpg SNR = ', SNR(img, co_4))


