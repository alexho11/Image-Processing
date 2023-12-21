import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import signal
from IP_ex3_func import sobel, non_max_suppresion, colorTransform, convolution



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

## Edge detection
# not needed for test images are already  binary images
image_current = image_noise
plt.figure()
plt.imshow(image_current)
# plt.imsave("../plots/test3.png", image_current, cmap='gray')
    
def hough_transform(img):
    height, width = img.shape
    max_d= int(np.ceil(np.sqrt(width**2 + height**2)))
    d = np.arange(-max_d, max_d,1)
    alpha = np.arange(-90, 90,1)
    alpha = np.deg2rad(alpha)
    accumulator = np.zeros((2 * max_d, len(alpha)), dtype=np.uint64)
    y_pixels, x_pixels = np.nonzero(img)  # (row, col) indexes to edges
    # loop over each pixel
    for i in range(len(x_pixels)):
        x=x_pixels[i]
        y=y_pixels[i]
        for idx_alpha in range(0, alpha.size):
            d_temp = int(np.round(x * np.cos(alpha[idx_alpha]) + y * np.sin(alpha[idx_alpha])))
            accumulator[d_temp, idx_alpha] += 1
    return accumulator, alpha, d

accumulator, alpha, d = hough_transform(image_current)
plt.figure()
plt.imshow(accumulator)


# Assuming you have 4 test images and their transformed images
# test_images = [image_ptsInRow, image_4pts, image_noise, image_lines]
# # Apply the transformation to each test image
# transformed_images = [hough_transform(image)[0] for image in test_images]
# fig, axs = plt.subplots(4, 2, figsize=(10, 20),sharey=True)  # 4 rows (for 4 images), 2 columns (for original and transformed)

# for i in range(4):
#     # Plot original image
#     axs[i, 0].imshow(test_images[i], cmap='gray')
#     axs[i, 0].set_title(f'Test Image {i+1}')
#     axs[i, 0].axis('off')  # Hide axes
#     axs[i, 0].set_aspect('equal')  # Set aspect ratio to be equal

#     # Plot transformed image
#     axs[i, 1].imshow(transformed_images[i], cmap='gray')
#     axs[i, 1].set_title(f'Transformed Image {i+1}')
#     axs[i, 1].axis('off')  # Hide axes
#     axs[i, 1].set_aspect('equal')  # Set aspect ratio to be equal

# plt.tight_layout()


# ## Initialization accumulator (2D histogram)
# alpha = np.arange(-90, 90,1)
# alpha = np.deg2rad(alpha)

# max_d= int(np.ceil(np.sqrt(40**2 + 40**2)))
# d = np.arange(-max_d, max_d,1)
# accumulator = np.zeros((2 * max_d, len(alpha)), dtype=np.uint64)

# ## Voting histogram using all edge pixels
# # get each edge pixel
# y_pixels, x_pixels = np.nonzero(image_current)  # (row, col) indexes to edges
# # loop over each pixel
# for i in range(len(x_pixels)):
#     x=x_pixels[i]
#     y=y_pixels[i]
#     for idx_alpha in range(0, alpha.size):
#         d_temp = int(np.round(x * np.cos(alpha[idx_alpha]) + y * np.sin(alpha[idx_alpha])))
#         accumulator[d_temp, idx_alpha] += 1




# accumulator, alpha, d = hough_transform(image_current)
# plt.figure()
# plt.imshow(accumulator)
# plt.imsave("../plots/test3_hough.png", accumulator, cmap='gray')
image_street = plt.imread("../images/building.jpg")
image_gray = cv2.cvtColor(image_street, cv2.COLOR_RGB2GRAY)
Sx,Sy,Sdir,Smag = sobel(image_gray)
plt.figure()
plt.imshow(Sdir)
plt.figure()
plt.imshow(Smag)

edges = non_max_suppresion(Smag, Sdir)

plt.figure()
plt.imshow(edges)

accumulator, alpha, d = hough_transform(edges)
      
plt.figure()
plt.imshow(accumulator)

plt.figure()
plt.imshow(image_gray,cmap='gray')
# draw the lines on the original image for over 100 votes
for i in range(accumulator.shape[0]):
    for j in range(accumulator.shape[1]):
        if accumulator[i,j] > 180:
            alpha_temp = alpha[j]
            d_temp = d[i]
            a=np.cos(alpha_temp)
            b=np.sin(alpha_temp)
            x0=a*d_temp
            y0=b*d_temp
            x1=int(x0+100*(-b))
            y1=int(y0+100*(a))
            x2=int(x0-100*(-b))
            y2=int(y0-100*(a))

            # Clip line coordinates to image boundaries
            x1 = np.clip(x1, 0, image_gray.shape[1] - 1)
            y1 = np.clip(y1, 0, image_gray.shape[0] - 1)
            x2 = np.clip(x2, 0, image_gray.shape[1] - 1)
            y2 = np.clip(y2, 0, image_gray.shape[0] - 1)

            plt.plot([x1,x2],[y1,y2],'r')

plt.show()
