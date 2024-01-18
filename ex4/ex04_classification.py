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

# Band 3 red, band 2 green, band 1 blue
# load image as rgb image
band3=plt.imread('./data/band3_train.tif')
band2=plt.imread('./data/band2_train.tif')
band1=plt.imread('./data/band1_train.tif')
# normalize bands to be in range [0, 1]
band3 = band3 / np.max(band3)
band2 = band2 / np.max(band2)
band1 = band1 / np.max(band1)
img_rgb_train=np.zeros((band1.shape[0],band1.shape[1],3))
img_rgb_train[:,:,0]=band3
img_rgb_train[:,:,1]=band2
img_rgb_train[:,:,2]=band1
# plt.figure()
# plt.imshow(img_rgb)
# plt.title('RGB image of traning data')
#plt.savefig('rgb_train.png')

# load test image as rgb image
band3=plt.imread('./data/band3_test.tif')
band2=plt.imread('./data/band2_test.tif')
band1=plt.imread('./data/band1_test.tif')
# normalize bands to be in range [0, 1]
band3 = band3 / np.max(band3)
band2 = band2 / np.max(band2)
band1 = band1 / np.max(band1)
img_rgb_test=np.zeros((band1.shape[0],band1.shape[1],3))
img_rgb_test[:,:,0]=band3
img_rgb_test[:,:,1]=band2
img_rgb_test[:,:,2]=band1
# plt.figure()
# plt.imshow(img_rgb)
# plt.title('RGB image of test data')
# plt.savefig('rgb_test.png')


# load label image (values from 1 to 7)
img_label_train = plt.imread('./data/labels_train.tif')
img_label_train_color = label2rgb(img_label_train,label_to_color)
plt.figure()
plt.subplot(122)
plt.imshow(img_label_train_color)
plt.title('Label of training data')
plt.subplot(121)
plt.imshow(img_rgb_train)
plt.title('RGB image of training data')
# plt.savefig('plots/train.png')

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

img_label_train_pred=y_pred.reshape(img_label_train.shape[0],img_label_train.shape[1])
img_label_train_pred_color=label2rgb(img_label_train_pred,label_to_color)
plt.figure()
plt.subplot(122)
plt.imshow(img_label_train_pred_color)
plt.title('Prediction result of training data')
plt.subplot(121)
plt.imshow(img_label_train_color)
plt.title('Label of training data')
# plt.savefig('plots/train_result.png')

# predict the test data

X_test=testdata.reshape(testdata.shape[0]*testdata.shape[1],testdata.shape[2])
y_test=img_label_test.reshape(img_label_test.shape[0]*img_label_test.shape[1])

y_pred_test=clf.predict(X_test)
img_label_test_pred=y_pred_test.reshape(testdata.shape[0],testdata.shape[1])
img_label_test_pred_color=label2rgb(img_label_test_pred,label_to_color)
print('Accuracy on test data: ', np.mean(y_pred_test==y_test))
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred_test)
print('Confusion matrix: ', cm)
print('sum1=', np.sum(cm, axis=1))
print('sum2=', np.sum(cm, axis=0))
print('sum3=', np.sum(cm))
# calculate the overall accuracy
OA = np.sum(np.diag(cm))/np.sum(cm)
# calculate the producer's accuracy and user's accuracy
# producer's accuracy
PA = np.zeros((7,1))
for ii in range(0,7):
    PA[ii] = cm[ii,ii]/np.sum(cm[ii,:])
print('Producer''s accuracy: ', PA)
# user's accuracy
UA = np.zeros((7,1))
for ii in range(0,7):
    UA[ii] = cm[ii,ii]/np.sum(cm[:,ii])
print('User''s accuracy: ', UA)
# kappa coefficient
p0 = OA
pe = np.sum(np.sum(cm, axis=1)*np.sum(cm, axis=0))/np.sum(cm)**2
kappa = (p0-pe)/(1-pe)
print('Kappa coefficient: ', kappa)
plt.figure()
plt.subplot(122)
plt.imshow(img_label_test_color)
plt.title('Label of test data')
plt.subplot(121)
plt.imshow(img_rgb_test)
plt.title('RGB image of test data')
# plt.savefig('plots/test.png')

plt.figure()
plt.subplot(122)
plt.imshow(img_label_test_pred_color)
plt.title('Prediction result of test data')
plt.subplot(121)
plt.imshow(img_label_test_color)
plt.title('Label of test data')
# plt.savefig('plots/test_result.png')

# predict only use rgb chanels
X_train_rgb = img_rgb_train.reshape(img_rgb_train.shape[0]*img_rgb_train.shape[1],img_rgb_train.shape[2])
X_test_rgb = img_rgb_test.reshape(img_rgb_test.shape[0]*img_rgb_test.shape[1],img_rgb_test.shape[2])
clf_rgb=GaussianNB()
clf_rgb.fit(X_train_rgb,y_train)
y_pred_rgb=clf_rgb.predict(X_test_rgb)
img_label_test_pred_rgb=y_pred_rgb.reshape(img_rgb_test.shape[0],img_rgb_test.shape[1])
img_label_test_pred_color_rgb=label2rgb(img_label_test_pred_rgb,label_to_color)
plt.figure()
plt.subplot(122)
plt.imshow(img_label_test_pred_color_rgb)
plt.title('Prediction result trained only on RGB')
plt.subplot(121)
plt.imshow(img_label_test_color)
plt.title('Label of test data')
plt.savefig('plots/test_result_rgb.png')
cm_rgb=confusion_matrix(y_test,y_pred_rgb)
print('Confusion matrix: ', cm_rgb)
print('sum1=', np.sum(cm_rgb, axis=1))
print('sum2=', np.sum(cm_rgb, axis=0))
print('sum3=', np.sum(cm_rgb))
# calculate the overall accuracy
OA_rgb = np.sum(np.diag(cm_rgb))/np.sum(cm_rgb)
print('Overall accuracy: ', OA_rgb)
# calculate the producer's accuracy and user's accuracy
# producer's accuracy
PA_rgb = np.zeros((7,1))
for ii in range(0,7):
    PA_rgb[ii] = cm_rgb[ii,ii]/np.sum(cm_rgb[ii,:])
print('Producer''s accuracy: ', PA_rgb)
# user's accuracy
UA_rgb = np.zeros((7,1))
for ii in range(0,7):
    UA_rgb[ii] = cm_rgb[ii,ii]/np.sum(cm_rgb[:,ii])
print('User''s accuracy: ', UA_rgb)
# plt.show()