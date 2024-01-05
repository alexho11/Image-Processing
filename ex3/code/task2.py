import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from hough import hough_lines_accumulator, hough_peaks
from IP_ex3_func import sobel, non_max_suppresion, colorTransform, convolution

img=cv.imread('../images/building.jpg')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
Sx,Sy,Sdir,Smag=sobel(gray)
plt.imsave('../plots/sobel_mag.png',Smag,cmap='gray')
plt.imsave('../plots/sobel_dir.png',Sdir,cmap='gray')
edges=non_max_suppresion(Sdir,Smag,size=30)
plt.imsave('../plots/edges_30.png',edges,cmap='gray')
# Hough Transform
accumulator, alpha, d = hough_lines_accumulator(edges)
peaks = hough_peaks(accumulator, num_peaks=5, threshold=40)

# Draw lines
for peak in peaks:
    rho = d[peak[0]]
    theta = alpha[peak[1]]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Save the result
cv.imwrite('../plots/houghlines_25_40.png', img)