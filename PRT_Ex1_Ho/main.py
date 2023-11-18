import os
import numpy as np
import matplotlib.pyplot as plt
import IP01_function as IP

# load the image 
plt.figure()
img = plt.imread('images/image.bmp')
#  print(img.shape)
#  print(img)
plt.imshow(img)

# calculate the histogram
hist = np.zeros(256)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        hist[img[i,j]] += 1

# plot the histogram
plt.figure()
plt.stairs(hist) 
plt.xlabel('Intensity')
plt.ylabel('Count')
# save the plot
#plt.savefig('histogram.png')
# check the histogram
plt.figure()
plt.stem(range(256), hist)  # Using plt.stem() instead of plt.hist()
plt.xlim([0,256])
plt.title('Histogram of the image')
plt.xlabel('Intensity')
plt.ylabel('Count')
# save the plot
#plt.savefig('histogram_check.png')

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
def cov(img1, img2):
    cov = np.zeros((img1.shape[0], img1.shape[1]))
    cov= np.sum((img1-np.mean(img1))*(img2-np.mean(img2)))/np.size(img1)
    return cov

img1=plt.imread('images/image_a.bmp')
img2=plt.imread('images/image_b.bmp')
print('cov = ', cov(img1, img2))
print('np.cov = ', np.cov(img1.reshape((1,-1)), img2.reshape((1,-1)))[0,1])

# calculate the correlation coefficient of the image
corr = cov(img1,img2)/(np.std(img1)*np.std(img2))
print('corr = ', corr)
print('np.corrcoef = ', np.corrcoef(img1.reshape((1,-1)), img2.reshape((1,-1)))[0,1])

def testImage(img):
    # calculate the histogram
    hist, bins = np.histogram(img.flatten(), 256, [0,256])
    # calculate the mean
    mean = np.mean(img)
    # calculate the variance
    var = np.var(img)
    # calculate the standard deviation
    std = np.std(img)
    return hist, mean, var, std

# load the images
test_img=['images/image_a.bmp', 'images/image_b.bmp', 'images/image_c.bmp', 'images/image_d.bmp']
a=['a', 'b', 'c', 'd']
for i in range(len(test_img)):
    img_t = plt.imread(test_img[i])
    hist, mean, var, std = testImage(img_t)
    print('Image ', test_img[i], ':')
    print('mean = ', mean)
    print('var = ', var)
    print('std = ', std)
    plt.figure()
    plt.stem(range(256), hist)
    plt.xlim([0,256])
    plt.title('Histogram of the image '+a[i])
    plt.xlabel('Intensity')
    plt.ylabel('Count')
    #plt.savefig('histogram_check'+str(i)+'.png')
for i in range(len(test_img)):
    img_t = plt.imread(test_img[i])
    sigma=cov(img, img_t)
    rho = sigma/(np.std(img)*np.std(img_t)) 
    print('Image ', test_img[i], ':')
    print('cov = ', sigma)
    print('corr = ', rho)

# Template search
# load the image
query = plt.imread('images/query.bmp')
def templateSearch(img, query):
    # return the most likely position of a template in the original image
    return IP.getMaximumCorrPoint(IP.correlation(img, query))
temp_list=['images/templateA.bmp', 'images/templateG.bmp', 'images/templateP.bmp', 'images/templateV.bmp']
alphabet = ['A', 'G', 'P', 'V']
for i in range(len(temp_list)):
    temp = plt.imread(temp_list[i])
    position = templateSearch(query, temp)
    print('templateSearch' +alphabet[i]+'= ', position)
    plt.figure()
    plt.imshow(IP.correlation(query, temp))
    plt.scatter(position[1], position[0], color='r') 
    plt.title('Correlation of the query and template'+alphabet[i])
    plt.savefig('correlation'+alphabet[i]+'.png')
    plt.figure()
    plt.imshow(query)
    plt.scatter(position[1], position[0], color='r')
    plt.title('Query with the template'+alphabet[i])
    plt.savefig('query'+alphabet[i]+'.png')

plt.show()