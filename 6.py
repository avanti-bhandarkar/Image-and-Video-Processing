import numpy as np
from scipy.fftpack import dct, idct
import random
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#Function to implement 2D DCT and IDCT
def dct2(a):
    return dct(dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return idct(idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

"""##DCT and IDCT on a random matrix

"""

# Generate a random integer matrix of size x * y 
x=8
y=8
f= np.random.randint(9,11,size=(x,y))
F= dct2(f)
print('original matrix',f,'\n')
print('DCT of the original matrix',F)

#Find Energy of the original image
fenergy= np.sum(f**2)
print('energy of the original image is', fenergy)

#Find Energy of the DCT of the image
Fenergy= np.sum(F**2)
print('energy of the DCT of the image is',Fenergy)

#Energy in the DC coefficient
print('energy in the DC coefficients only is',F[0,0]**2)

#Retain coefficients with energy more than x% of the energy of the Total energy
percent = float(input('enter a percentage of energy you want to retain '))
P= percent*Fenergy
print('if only',percent,'% of the energy is retained, then energy becomes',P)

Fnew= np.where(F**2 > P, F,0) #set values of matrix to 0 if the square value of the DCT matrix is greater than the retained value threshold
print('new DCT matrix',Fnew)

# Finding the nonzero values in the compressed matrix
ind,val= np.unique(Fnew,return_counts=True)
nonzeros = np.sum(val[np.where(ind!=0)])
print('number of nonzero values in the matrix is/are',nonzeros)

#Compression Ratio
Original = x*y
Compressed = nonzeros
CR= Original/Compressed
print('compression ratio of the image is',CR)

#retrieving the estimate of the original matrix
fnew= idct2(Fnew)
fnew= fnew.astype(int)
print('performing inverse DCT we get the following matrix \n', fnew)

#Obtain Mean Square Error between f and fnew
mse= np.sum(((f-fnew)**2))/(x*y)
print('mean square error between the original matrix and the matrix obtained from idct is',mse)

# Retain first row of the transformed matrix
Fnew1= np.zeros((x,y), dtype =float)
Fnew1[0,:]= F[0,:]
print(Fnew1)

fnew1= idct2(Fnew1)
fnew1= fnew1.astype(int)
print(fnew1)

# Retain first row and first column of the transformed matrix
Fnew2= np.zeros((x,y), dtype =float)
Fnew2[0,:]= F[0,:]
Fnew2[:,0]= F[:,0]
print(Fnew2)

fnew2= idct2(Fnew2)
fnew2= fnew2.astype(int)
print(fnew2)

#Retain the first four coefficients
Fnew3= np.zeros((x,y), dtype =float)
Fnew3[0:2,0:2]= F[0:2,0:2]
print(Fnew3)

fnew3= idct2(Fnew3)
fnew3= fnew3.astype(int)
print(fnew3)

"""##DCT for an image with and without compression"""

img = cv2.imread('/content/lake.tif',0)  # Read the image
m,n= img.shape #Find and store size of image in m,n
N= int(input('Enter block size ')) #Size of the block
percent = float(input('Enter retention percentage ')) #enter percentage of enegry to be retained
plt.imshow(img,cmap='gray') #display original image
plt.title('Original image')
plt.show()

#Find DCT of the given image using DCT function defined before
imgdct = np.zeros((m,n), dtype=int) #create matrix of zeros with same size as original image
for row in range(m//N): #traverse rows
        for col in range(n//N): #traverse columns
               imgdct[row*N:(row+1)*N,col*N:(col+1)*N]= dct2(img[row*N:(row+1)*N,col*N:(col+1)*N])

plt.imshow(imgdct, cmap="gray") #display image after dct 
plt.title('Image after dct with block size %i'%N) 
plt.show()

#Total energy of the image
energy= np.sum(img**2)
print('Total energy of the image is ',energy)

#For compression, retaining only P percent coefficients with max magnitude. 
P= percent*energy
Inew= np.where(imgdct**2 > P, imgdct,0)

# Finding the nonzero values in the compressed matrix
ind,val= np.unique(Inew,return_counts=True)
nonzeros = np.sum(val[np.where(ind!=0)])
print('Number of nonzero values in the image is' ,nonzeros)

#Compression Ratio
CR= (m*n)/nonzeros #compression ratio is the number of pixels of original image by retained coefficients
print("Total coefficient in the input image is ", m*n)
print("Total coefficients retained in the output image is ", nonzeros)
print("Compression ratio = input size/output size = ", CR)

#Find IDCT of the given image  using IDCT function defined before
imgidct = np.zeros((m,n), dtype=int) #create matrix of zeros with same size as original image
for row in range(m//N): #traverse rows
        for col in range(n//N): #traverse columns
              imgidct[row*N:(row+1)*N,col*N:(col+1)*N]= idct2(Inew[row*N:(row+1)*N,col*N:(col+1)*N])
plt.imshow(imgidct,cmap="gray") #display compressed image
plt.title('Compressed image') 
plt.show()

#MSE
mse = mean_squared_error(imgidct,img)
print('Mean Squared Error is ',mse)

from google.colab.patches import cv2_imshow
cv2_imshow(imgdct)
cv2_imshow(imgidct)

"""##alternative method for dct"""

#alternative method for finding nonzero values in the compressed matrix
nonzeros= np.sum([Inew!=0])
print(nonzeros)

from math import cos, pi, sqrt
import numpy as np

def dct1D(image, num):
    n = len(image)
    dctImage= np.zeros_like(image).astype(float)
    for k in range(n):
        sum = 0
        for i in range(n):
            sum += image[i] * cos(2 * pi * k / (2.0 * n) * i + (k * pi) / (2.0 * n))
        ck = sqrt(0.5) if k == 0 else 1
        dctImage[k] = sqrt(2.0 / n) * ck * sum
#saving the N largest numbers and resetting all the others
    if num > 0:
        dctImage.sort()
        for i in range(num, n):
            dctImage[i] = 0

    return dctImage

def dct2D(image, num):

    height = image.shape[0]
    width = image.shape[1]
    temp = np.zeros_like(image).astype(float)
    dct_matrix= np.zeros_like(image).astype(float)

    for h in range(height):
        temp[h, :] = dct1D(image[h, :], num)
    for w in range(width):
        dct_matrix[:, w] = dct1D(temp[:, w], num)
    return dct_matrix

def idct1D(image):

    n = len(image)
    idctImage = np.zeros_like(image).astype(float)
    for i in range(n):
        sum = 0
        for k in range(n):
            ck = sqrt(0.5) if k == 0 else 1 
            sum += ck * image[k] * cos(2 * pi * k / (2.0 * n) * i + (k * pi) / (2.0 * n))

        idctImage[i] = sqrt(2.0 / n) * sum
    return idctImage

def idct2D(image):
    height = image.shape[0]
    width =  image.shape[1]

    temp = np.zeros_like(image).astype(float)
    idct_matrix = np.zeros_like(image).astype(float)

    for h in range(height):
        temp[h, :] = idct_1D(image[h, :])
    for w in range(width):
        idct_matrix[:, w] = idct_1D(temp[:, w])
    return idct_matrix

import cv2

img = cv2.imread('/content/jetplane.tif',0)  # Read the image
Coef = 32

print("DCT")
imgResult = dct2D(img,Coef)
plt.imshow(imgResult)

print("Inverse DCT")
idct_img = idct2D(imgResult)
plt.imshow(idct_img)
