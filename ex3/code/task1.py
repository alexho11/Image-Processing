import matplotlib.pyplot as plt
import numpy as np
import cv2
from hough import hough_lines_accumulator, hough_peaks


# test points image 02
image_4pts = np.zeros((40,40))
image_4pts[20,7] = 1
image_4pts[7,20] = 1
image_4pts[20,33] = 1
image_4pts[33,20] = 1

# test points image 01
image_ptsInRow = np.zeros((40,40))
image_ptsInRow[5,0] = 1
image_ptsInRow[15,10] = 1
image_ptsInRow[25,20] = 1
image_ptsInRow[35,30] = 1

img=image_4pts
accumulator, thetas, rhos = hough_lines_accumulator(img)

plt.figure()
plt.imshow(img,cmap='gray')
plt.figure()
plt.imshow(accumulator)
accumulator_copy = accumulator.copy()
peaks = hough_peaks(accumulator_copy, 20,1)


d_idx, theta_idx = peaks[8]
d= rhos[d_idx]
alpha = thetas[theta_idx]
print('Line 1: d=',d,'alpha=',alpha/np.pi*180,'accumulated votes=',accumulator[93,135])
a=np.cos(alpha)
b=np.sin(alpha)
x0=a*d  
y0=b*d
x1=int(x0+1000*(-b))
y1=int(y0+1000*(a))
x2=int(x0-1000*(-b))
y2=int(y0-1000*(a))
cv2.line(img,(x1,y1),(x2,y2),(255,255,255),1)
d_idx, theta_idx = peaks[14]
d= rhos[d_idx]
alpha = thetas[theta_idx]
print('Line 2: d=',d,'alpha=',alpha/np.pi*180, 'accumulated votes=',accumulator[d_idx,theta_idx])
a=np.cos(alpha)
b=np.sin(alpha)
x0=a*d  
y0=b*d
x1=int(x0+1000*(-b))
y1=int(y0+1000*(a))
x2=int(x0-1000*(-b))
y2=int(y0-1000*(a))
cv2.line(img,(x1,y1),(x2,y2),(255,255,255),1)
plt.figure()
plt.imshow(img,cmap='gray')
plt.show()