import matplotlib.pyplot as plt
import numpy as np


# Task 2 - Mean and Median

def median(listValues):
    median_value = 0.0
    if len(listValues)%2 == 0:
        raise ValueError('Number of values must be odd')
    else:
        sorted_listValues = np.sort(listValues)
        median_value = sorted_listValues[int(len(listValues)/2)]
     
    return median_value

# 1D median & mean
values = [19.7, 556.3, 23.2, 27.5, 16.3, 21.0, 27.2, 495.0, 25.3]
print('Mean: ', np.mean(values))
print('Median: ', median(values))

# 2D median & mean

def addBorder_mirror(img, br, bc):
    row,col = img.shape
    imgOut = np.zeros((row +2*br,col+2*bc),np.uint8)
    r = 0
    c = 0
    for px in np.nditer(imgOut[:,:], op_flags=["readwrite"]):
        rI = br - r
        cI = bc - c
        if rI < 0:
            rI = r - br 
        if rI >= row:
            rI = row - (rI - row) - 2
        if cI < 0:
            cI = c - bc
        if cI >= col:
            cI = col - (cI - col) - 2
        px[...] = img[rI,cI]
        c+=1
        if c >= imgOut.shape[1]:
            c=0
            r+=1
    return imgOut


def convolution(img,kernel):  
    rows,cols = kernel.shape
    n = float(rows*cols)
    rI,cI = img.shape
    imgOut = np.zeros((rI,cI),np.float32)
    startC = int(cols/2 )
    startR = int(rows/2 )
    imgBorder = addBorder_mirror(img, startR, startC)
    imgBorder.astype(np.float32)
    r = 0
    c = 0
    for pxOut in np.nditer(imgOut[:,:], op_flags =["writeonly"]):
        it = np.nditer([imgBorder[r : r+2*startR+1 , c : c+2*startC+1],kernel[:,:]],flags=["buffered","external_loop"],op_flags =["readonly"], op_dtypes=["float64","float64"])
        val = 0.0
        for i,k in it:
            val += np.sum(i*k)
        pxOut[...] = val
        c+= 1
        if c >= imgOut.shape[1]:
            c=0
            r+=1
    return imgOut


def medianImg(img,size):
    imgB = addBorder_mirror(img,int(size/2),int(size/2))
    imgOut = np.zeros(img.shape,img.dtype)
    c=0
    r=0
    for pxOut in np.nditer(imgOut[:,:], op_flags =["writeonly"]):
        it = np.nditer(imgB[r : r+size , c : c+size],flags=["buffered","external_loop"],op_flags =["readonly"])
        for x in it:
           # print(x)
            pxOut[...] = median(x)
        c+=1
        if c >= imgOut.shape[1]:
            c=0
            r+=1
    return imgOut


chessboard = np.array([[0,0,0,0,0,255,255,255,255,255],[0,0,0,0,0,255,255,255,255,255],
                       [0,0,0,0,0,255,255,255,255,255],[0,0,0,0,0,255,255,255,255,255],
                       [0,0,0,0,0,255,255,255,255,255],[255,255,255,255,255,0,0,0,0,0],
                       [255,255,255,255,255,0,0,0,0,0],[255,255,255,255,255,0,0,0,0,0],
                       [255,255,255,255,255,0,0,0,0,0],[255,255,255,255,255,0,0,0,0,0]])

chessboard_noisy = np.loadtxt("data/chessboard_noisy.txt")

chessboard_median = medianImg(chessboard,3)
chessboard_mean = convolution(chessboard,np.ones((3,3),np.float32)/9)
chessboard_noisy_median = medianImg(chessboard_noisy,3)
chessboard_noisy_mean = convolution(chessboard_noisy,np.ones((3,3),np.float32)/9)
figure, axes = plt.subplots(3, 2)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
axes[0,0].imshow(chessboard, cmap='gray')
axes[0,0].set_title("Original")
axes[1,0].imshow(chessboard_median, cmap='gray')
axes[1,0].set_title("Median")
axes[2,0].imshow(chessboard_mean, cmap='gray')
axes[2,0].set_title("Mean")
axes[0,1].imshow(chessboard_noisy, cmap='gray')
axes[0,1].set_title("Noisy")
axes[1,1].imshow(chessboard_noisy_median, cmap='gray')
axes[1,1].set_title("Median noisy")
axes[2,1].imshow(chessboard_noisy_mean, cmap='gray')
axes[2,1].set_title("Mean noisy")
#plt.show()
#plt.savefig("chessboard.png")
# Task 3 - Sobel and Laplace

def sobel(img):
    ksx = np.zeros((3,3),np.float32)   
    ksx[:]= [[-1,0,1],[-2,0,2],[-1,0,1]]
    ksy = np.zeros((3,3),np.float32)
    ksy[:]= [[-1,-2,-1],[0,0,0],[1,2,1]]
    Sx = convolution(img, ksx)
    Sy = convolution(img, ksy)
    return Sx,Sy 

def laplace(img):
    kl = np.zeros((3,3),np.float32)
    kl[:]= [[0,-1,0],[-1,4,-1],[0,-1,0]]
    return convolution(img,kl)

img=plt.imread("data/old_town.jpg")
Sx,Sy = sobel(img)
Smag = np.sqrt(Sx**2 + Sy**2)
Sdir = np.arctan2(Sy,Sx)
L = laplace(img)
figure, axes = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
axes[0,0].imshow(Sx, cmap='gray')
axes[0,0].set_title("Sobel x")
axes[1,0].imshow(Sy, cmap='gray')
axes[1,0].set_title("Sobel y")
axes[0,1].imshow(Smag, cmap='gray')
axes[0,1].set_title("Sobel magnitude")
axes[1,1].imshow(Sdir, cmap='gray')
axes[1,1].set_title("Sobel direction")
#plt.savefig("sobel.png")

plt.figure()
laplace_img = laplace(img)
plt.imshow(laplace_img, cmap='gray')
plt.title("Laplace")
##plt.savefig("laplace.png")
#plt.show()

