import os
import numpy as np
import matplotlib.pyplot as plt
import IP01_function as IP

# load the image 
plt.figure()
img = plt.imread('images/image.bmp')
#  print(img.shape)
print(img)
plt.imshow(img)

# calculate the histogram
hist = np.zeros(256)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        hist[img[i,j]] += 1

# plot the histogram
plt.figure()
plt.stairs(hist)


# calculate the mean of the image
mean = np.sum(img)/np.size(img)
print('mean = ', mean)
print('np.mean = ', np.mean(img))

# calculate the variance of the image
var = np.sum((img-mean)**2)/np.size(img)
print('var = ', var)
print('np.var = ', np.var(img))

# calculate the standard deviation of the image
std = np.sqrt(var)
print('std = ', std)
print('np.std = ', np.std(img))

# calculate the covariance of the image
def cov_matrix(img1, img2):
    cov = np.zeros((img1.shape[0], img1.shape[1]))
    cov= np.sum((img1-np.mean(img1))*(img2-np.mean(img2)))/np.size(img1)
    return cov

img1=plt.imread('images/image_a.bmp')
img2=plt.imread('images/image_b.bmp')
cov=cov_matrix(img1, img2)
print('cov = ', cov)
print('np.cov = ', np.cov(img1, img2))

# calculate the correlation coefficient of the image
corr = cov/(np.std(img1)*np.std(img2))
print('corr = ', corr)
print('np.corrcoef = ', np.corrcoef(img1, img2))
# plt.show()