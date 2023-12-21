
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import signal
from IP_ex3_func import sobel, non_max_suppresion, colorTransform, convolution


#%% 1.

# test points image 01
image_ptsInRow = np.zeros((40,40))
image_ptsInRow[5,0] = 1
image_ptsInRow[15,10] = 1
image_ptsInRow[25,20] = 1
image_ptsInRow[35,30] = 1

# test points image 02
image_4pts = np.zeros((40,40))
image_4pts[20,7] = 1
image_4pts[7,20] = 1
image_4pts[20,33] = 1
image_4pts[33,20] = 1

# test points image 03
image_noise = np.zeros((40,40))
image_noise[np.random.randint(37, size=(23,1)),np.random.randint(37, size=(23,1))] = 1

# test lines image 04
image_lines = np.zeros((40,40))
image_lines[:,15] = 1
for ii in range(1,40,1):
    image_lines[ii,-ii] = 1

#%%
## Edge detection
# not needed for test images are already  binary images
image_current = image_ptsInRow
plt.imshow(image_current)

## Initialization accumulator (2D histogram)
#TODO
alpha = ... #TODO
alpha = np.deg2rad(alpha)

## Voting histogram using all edge pixels
# get each edge pixel
y_pixels, x_pixels = np.nonzero(image_current)  # (row, col) indexes to edges
# loop over each pixel
for i in range(len(x_pixels)):
    #TODO
    for idx_alpha in range(0, alpha.size):
        #TODO
        accumulator[d, idx_alpha] += 1
      

# plt.imshow(accumulator)

#%% 2.

image_street = plt.imread("..\\images\\building.jpg")
image_gray = cv2.cvtColor(image_street, cv2.COLOR_RGB2GRAY)
# TODO
