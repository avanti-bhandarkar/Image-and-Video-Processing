
#import libraries
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

#Digital negative
img= cv2.imread('/content/Fig0338(a)(blurry_moon).tif',0) #read image
m,n= img.shape    #find dimensions of the shape                 
L= img.max() #find maximum pixel value in the image                      
negative = L-img   #apply negative transform to the entire array              
cv2.imwrite('moon_negative.png', negative) #save negative image as png

#plot both the images side-by-side to see the difference
plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.title('Original image')
plt.imshow(img,cmap="gray")
plt.subplot(1,2,2)
plt.title('Image obtained after digital negative transformation')
plt.imshow(negative,cmap='gray')

#Thresholding  
img1= cv2.imread('/content/Fig0310(b)(washed_out_pollen_image).tif',0) #read image
T = int(input('Enter a threshold value: ')) #Take threshold value as input from the user and convert to integer
m,n= img1.shape                 #find dimensions of the shape      
thresh= np.zeros((m,n), dtype=int) # create an array of zeros
for i in range(m):
    for j in range(n):
        if img1[i,j]< T: #if pixel value in image is less than threshold value, floor to 0
            thresh[i,j]= 0
        else:
            thresh[i,j] = 255 #if pixel value in image is more than threshold value, ceil to 255

cv2.imwrite('pollen_thresh.png', thresh) #save image as png

#plot both the images side-by-side to see the difference
plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.title('Original image')
plt.imshow(img1,cmap='gray')
plt.subplot(1,2,2)
plt.title('Image obtained after thresholding with r1 = %i'%T)
plt.imshow(thresh,cmap='gray')

#Thresholding  
img1= cv2.imread('/content/Fig0310(b)(washed_out_pollen_image).tif',0) #read image

T1 = 100
m,n= img1.shape                 #find dimensions of the shape      
thresh= np.zeros((m,n), dtype=int) # create an array of zeros
for i in range(m):
    for j in range(n):
        if img1[i,j]< T1: #if pixel value in image is less than threshold value, floor to 0
            thresh[i,j]= 0
        else:
            thresh[i,j] = 255 #if pixel value in image is more than threshold value, ceil to 255
T2 =125
p,q= img1.shape                 #find dimensions of the shape      
thresh2= np.zeros((p,q), dtype=int) # create an array of zeros
for i in range(p):
    for j in range(q):
        if img1[i,j]< T2: #if pixel value in image is less than threshold value, floor to 0
            thresh2[i,j]= 0
        else:
            thresh2[i,j] = 255 #if pixel value in image is more than threshold value, ceil to 255

#plot both the images side-by-side to see the difference
plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.title('Image obtained after thresholding with r1 = %i'%T1)
plt.imshow(thresh,cmap='gray')
plt.subplot(1,2,2)
plt.title('Image obtained after thresholding with r1 = %i'%T2)
plt.imshow(thresh2,cmap='gray')

#Grey level slicing with and without background 

img3 = cv2.imread('/content/Fig0312(a)(kidney).tif',0) #read image
m,n= img3.shape #find dimensions of the shape  
T1= int(input('Enter the lower threshold value T1:')) #Take the lower threshold value as user input and convert to integer
T2= int(input('Enter the upper threshold value T2:')) #Take the upper threshold value as user input and convert to integer
gls_withback= np.zeros((m,n), dtype=int) # create a array of zeros
gls_withoutback= np.zeros((m,n), dtype=int) # create a array of zeros

#GLS with background
for i in range(m):
    for j in range(n):
        if T1 < img3[i,j] < T2: #check if a pixel lies between the threshold
            gls_withback[i,j]= 255 #if it lies in the threshold range then make the pixel light
        else:
            gls_withback[i,j] = img3[i,j] #if it does not lie in the threshold range, retain colour from original image

#GLS without background
for i in range(m):
    for j in range(n):
        if T1 < img3[i,j] < T2: #check if a pixel lies between the threshold
            gls_withoutback[i,j]= 255 #if it lies in the threshold range then retain colour from original image
        else:
            gls_withoutback[i,j] = 0 #if it does not lie in the threshold range, then make the pixel black

#plot both the images side-by-side to see the difference
plt.figure(figsize=(10,10))
plt.subplot(1,3,1)
plt.title('Original image')
plt.imshow(img3,cmap='gray')
plt.subplot(1,3,2)
plt.title('GLS with background')
plt.imshow(gls_withback,cmap='gray')
plt.subplot(1,3,3)
plt.title('GLS without background')
plt.imshow(gls_withoutback,cmap='gray')

#Contrast stretching

img4 = cv2.imread('/content/Fig0310(b)(washed_out_pollen_image).tif',0) #read image
m,n = img4.shape #find dimensions of the shape  
cs= np.zeros((m,n), dtype=int) # create an array of zeros
r1 = int(input('Enter a lower threshold for the original image: ')) #take user input for lower threshold value
s1 = 0 #alternatively img4.min()
r2= int(input('Enter an upper threshold for the original image: ')) #take user input for upper threshold value
s2 = 255 #alternatively img4.max()

for i in range(m):
    for j in range(n):
      if (0 <= img4[i,j] and img4[i,j] < r1): #for first slope the value of r is between 0 and r1
        cs[i,j]= (s1 / r1)*img4[i,j] #pixel value in modified image is s=mr where m = s1/r1
      elif (r1 <= img4[i,j] and img4[i,j] < r2): #for second slope the value of r is between r1 and r2
        cs[i,j] = ((s2 - s1)/(r2 - r1)) * (img4[i,j] - r1) + s1 #pixel value in modified image is s=m(r-r1)+s1 where m = (s2 - s1)/(r2 - r1)
      else:  #for second slope the value of r is between r2 and L-1
        cs[i,j]=((255 - s2)/(255 - r2)) * (img4[i,j] - r2) + s2 #pixel value in modified image is s=m(r2-r1)+s2 where m = (L-1 - s2)/(L-1 - r2)

#plot both the images side-by-side to see the difference
plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.title('Original image')
plt.imshow(img4,cmap='gray')
plt.subplot(1,2,2)
plt.title('Image obtained after contrast stretching')
plt.imshow(cs,cmap='gray')
