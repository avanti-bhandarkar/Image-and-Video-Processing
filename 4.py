#import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

#low pass filter with hard coding

img = cv2.imread('1.tif',0) #load the image
m,n=img.shape #obtain number of rows and columns of the image
mask= np.array([[1,1,1],[1,1,1],[1,1,1]]) #define mask for low pass filtering
#mask = np.ones([3,3],dtype=int) alternative for above line
mask = mask/9 #complete defining the mask

low_pass=np.zeros([m,n]) #define a matrix of zeroes with same shape as the original image
for i in range(1,m-1): #traverse through rows
    for j in range(1,n-1): #traverse through columns
        low_pass[i,j] = img[i-1,j-1]*mask[0,0]+img[i-1,j]*mask[0,1]+img[i-1,j+1]*mask[0,2]+img[i,j-1]*mask[1,0]+img[i,j]*mask[1,1]+img[i,j+1]*mask[1,2]+img[i+1,j-1]*mask[2,0]+img[i+1,j]*mask[2,1]+img[i+1,j+1]*mask[2,2]

plt.figure(figsize=(15,15))       # define figure size
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=1) # add spaces between subplots
plt.subplot(2,2,1)
plt.imshow(img,cmap='gray',vmin = 0, vmax = 255) #show original image
plt.title('Original image')      #display original image
plt.axis('off')
plt.subplot(2,2,2)        
plt.imshow(low_pass,cmap='gray',vmin = 0, vmax = 255) #show low-pass filtered image with 3x3 mask             
plt.title('Low Pass Filtered Image filtered with 3x3 LPF')  
plt.axis('off')
plt.show()

#low pass filter 

img = cv2.imread('1.tif',0) #load the image
m,n=img.shape

plt.figure(figsize=(60,60)) #define figure sizes
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=2, hspace=2) 
plt.subplot(4 ,4 ,1)
plt.imshow(img,cmap='gray'),plt.title ('Original Image') #plot original image
plt.axis('off') 

ws = int(input('Enter the desired window size (odd values only) for the filter: ')) #take user input for window size
ws1 = ws//2 #divide window size
img_new = np.zeros([m,n],dtype=int) #define a matrix of zeroes with same shape as the original image
for i in range(1,m-1): #traverse through rows
  for j in range(1,n-1): #traverse through columns
    temp = img[i-ws1:i+ws1,j-ws1:j+ws1]   
    total = np.sum(temp)
    img_new[i,j] = total//ws1**2

plt.subplot(4,4,2)
plt.imshow(img_new,cmap='gray')
plt.title('Low Pass Filtered Image with Filter size of %i' %ws)
plt.axis('off')

ws = int(input('Enter the desired window size (odd values only) for the filter: ')) #take user input for window size
ws1 = ws//2 #divide window size
img_new = np.zeros([m,n],dtype=int) #define a matrix of zeroes with same shape as the original image
for i in range(1,m-1): #traverse through rows
  for j in range(1,n-1): #traverse through columns
    temp = img[i-ws1:i+ws1,j-ws1:j+ws1]   
    total = np.sum(temp)
    img_new[i,j] = total//ws1**2

plt.subplot(4,4,3)
plt.imshow(img_new,cmap='gray')
plt.title('Low Pass Filtered Image with Filter size of %i' %ws)
plt.axis('off')

ws = int(input('Enter the desired window size (odd values only) for the filter: ')) #take user input for window size
ws1 = ws//2 #divide window size
img_new = np.zeros([m,n],dtype=int) #define a matrix of zeroes with same shape as the original image
for i in range(1,m-1): #traverse through rows
  for j in range(1,n-1): #traverse through columns
    temp = img[i-ws1:i+ws1,j-ws1:j+ws1]   
    total = np.sum(temp)
    img_new[i,j] = total//ws1**2

plt.subplot(4,4,4)
plt.imshow(img_new,cmap='gray')
plt.title('Low Pass Filtered Image with Filter size of %i' %ws)
plt.axis('off')

img = cv2.imread('1.tif',0) #load the image

plt.subplot(2,2,1)
plt.imshow(img,cmap='gray'),plt.title ('Original Image')
plt.axis('off')
ws=int(input('Enter the desired window size (odd values only) for the filter: ')) #take user input for window size

new=np.ones([ws,ws])
pd=int((ws-1)/2)
start=ws-pd-1
row,col=img.shape
f=np.pad(img,[(pd,pd)],'constant')
r,c=f.shape;
lpf=np.zeros([row,col])
for i in range(start,r-pd):
    for j in range(start,c-pd):
        window=f[(i-pd):(i+pd+1),(j-pd):(j+pd+1)];
        su=0;        
        for s in range(ws):
            for t in range(ws):
                m=np.multiply(new[s][t],window[s][t])
                su=np.add(su,m)
        lpf[i-start,j-start]=su;
out1=lpf/9
plt.subplot(2,2,2)
plt.imshow(np.uint8(out1),cmap='gray'),plt.title ('Low Pass Filtered Image')
plt.axis('off')

# Median Spatial Domain Filtering 
  
  
import cv2 
import numpy as np 
  
# Read the image 
img_noisy1 = cv2.imread('noisysalterpepper.png', 0) 
  
# Obtain the number of rows and columns  
# of the image 
m, n = img_noisy1.shape 
   
# Traverse the image. For every 3X3 area,  
# find the median of the pixels and 
# replace the ceter pixel by the median 
img_new1 = np.zeros([m, n]) 
  
for i in range(1, m-1): 
    for j in range(1, n-1): 
        temp = [img_noisy1[i-1, j-1], 
               img_noisy1[i-1, j], 
               img_noisy1[i-1, j + 1], 
               img_noisy1[i, j-1], 
               img_noisy1[i, j], 
               img_noisy1[i, j + 1], 
               img_noisy1[i + 1, j-1], 
               img_noisy1[i + 1, j], 
               img_noisy1[i + 1, j + 1]] 
          
        temp = sorted(temp) 
        img_new1[i, j]= temp[4] 
  
img_new1 = img_new1.astype(np.uint8) 
from google.colab.patches import cv2_imshow
cv2_imshow(img_noisy1)
cv2_imshow(img_new1)

newimg = np.zeros([m, n]) 
mask = int(input('Enter mask size'))
mask1 = mask//2
for i in range(1,m-1): #traverse through rows
  for j in range(1,n-1): #traverse through columns
      temp = img_noisy1[i-ws1:i+ws1,j-ws1:j+ws1]   
      filter = np.median(temp)
      newimg[i,j]=filter

cv2_imshow(img_noisy1)
cv2_imshow(newimg)
