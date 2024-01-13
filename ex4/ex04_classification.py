# -*- coding: utf-8 -*-
"""
Spectral channels
1 0.45-0.52 µm, blue-green
2 0.52-0.60 µm, green
3 0.63-0.69 µm, red
4 0.76-0.90 µm, near infrared
5 1.55-1.75 µm, mid infrared
6 10.4-12.5 µm (60 × 60 m) Thermal channel
7 2.08-2.35 µm, mid infrared

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB


label_to_color = {
    1: [0,  0, 128], # water
    2: [0,  128, 0], # forest
    3: [ 0,  255,  0], # vegetation
    4: [0, 221, 221], # ice
    5: [255, 255, 255], # snow
    6: [255, 0, 0], # rock
    7: [80, 80, 80] # shadow
}

# convert one channel label image [nxm] to a given colormap 
# resutling in a rgb image [nxmx3]
def label2rgb(img_label, label_to_color):
    h, w = img_label.shape
    img_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for gray, rgb in label_to_color.items():
        img_rgb[img_label == gray, :] = rgb
    return img_rgb

# load label image (values from 1 to 7)
img_label_train = plt.imread('./data/labels_train.tif')
img_label_train_color = label2rgb(img_label_train,label_to_color)
plt.imshow(img_label_train_color)

# load traindata (7 bands)
traindata = np.zeros((img_label_train.shape[0],img_label_train.shape[1],7))
for ii in range(0,7):
    I = plt.imread('./data/band' + str(ii+1) + '_train.tif')
    traindata[:,:,ii] = I
img_label_test = plt.imread('./data/labels_test.tif')
img_label_test_color = label2rgb(img_label_test,label_to_color)
# load testdata (7 bands)
testdata = np.zeros((img_label_test.shape[0],img_label_test.shape[1],7))
for ii in range(0,7):
    I = plt.imread('./data/band' + str(ii+1) + '_test.tif')
    testdata[:,:,ii] = I

# flatten the train data and labels
X_train = traindata.reshape(traindata.shape[0]*traindata.shape[1],traindata.shape[2])
y_train = img_label_train.reshape(img_label_train.shape[0]*img_label_train.shape[1])

clf=GaussianNB()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_train)
print('Accuracy on training data: ', np.mean(y_pred==y_train))

# predict the test data

X_test=testdata.reshape(testdata.shape[0]*testdata.shape[1],testdata.shape[2])
y_test=img_label_test.reshape(img_label_test.shape[0]*img_label_test.shape[1])
y_pred_test=clf.predict(X_test)
img_label_test=y_pred_test.reshape(testdata.shape[0],testdata.shape[1])
img_label_test_color=label2rgb(img_label_test,label_to_color)
print('Accuracy on test data: ', np.mean(y_pred_test==y_test))
plt.figure()
plt.imshow(img_label_test_color)
plt.title('Classification result')
plt.show()
# plt.show()