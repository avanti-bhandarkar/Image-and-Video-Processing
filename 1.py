
#import libraries

import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

img= cv2.imread("/content/jetplane.tif",0)
print('Image type is')
print(type(img))
cv2_imshow(img)

print('Original shape is ',img.shape)
[m,n]=img.shape

"""### *Downsampling*"""

#downsampling rate
ds = int(input('Enter a downsampling rate: '))

#downsampling img1 by ds

img1 = np.zeros((m//ds,n//ds), dtype = np.int)

for i in range (0,m,ds):
  for j in range (0,n,ds):
    try:
      img1[i//ds][j//ds]= img[i][j]
    except IndexError:
      pass

print('Old shape was ',img.shape,'\n')
print('New shape is ',img1.shape)
cv2.imwrite('downsample.png',img1)
print('\nDownsampled by factor of',ds,'\n')
cv2_imshow(img1)


###Upsampling


#upsampling rate
us = int(input('Enter an upsampling rate: '))

#upsampling img1 by 2 to get back img (here img2)
down =  cv2.imread("/content/downsample.png",0)

img2 = np.zeros((m,n), dtype = np.int)

for i in range (0,m-1,us):
  for j in range (0,n-1,us):
      img2[i,j]=down[i//ds][j//ds]

print('Old shape was ',down.shape,'\n')
print('New shape is ',img2.shape)
cv2.imwrite('upsample.png',img2)
print('\nRestored original image by upsampling by factor of',us,'\n')
cv2_imshow(img2)


#NN interpolation - replication 

#column replication
colrep = cv2.imread("/content/jetplane.tif",0)
us = 2
for i in range(0,m-1):
    for j in range(1,n-1,us):
        colrep[i,j]= colrep[i,j-1]
#print('\nShape is ',colrep.shape,'\n')

#row replication
rowrep =  cv2.imread("/content/jetplane.tif",0)
us = 2
for i in range(0,m-1):
    for j in range(1,n-1,us):
        rowrep[i,j]= rowrep[i,j-1]
#print('\nShape is ',rowrep.shape,'\n')

plt.figure(figsize=(20,20))
plt.subplot(1,2,1)
plt.title('Column-replicated image')
plt.imshow(colrep,cmap="gray")
plt.subplot(1,2,2)
plt.title('Row-replicated image')
plt.imshow(rowrep,cmap='gray')


#upsampling img1 by 2 to get back img (tried on my own)

[r,c] = np.shape(down)
[r1,c1] = [int(np.ceil(r * us)), int(np.ceil(c * us))]

down1 = np.zeros((r1,c1), dtype=int)
for i in range(r1):
    for j in range(c1):
        x = int(np.floor(i / us))
        y = int(np.floor(j / us))
        down1[i, j] = down[x, y]

print('Old shape was ',down.shape,'\n')

print('New shape is ',down1.shape)
cv2.imwrite('upsample.png',down1)
print('\nRestored original image by upsampling by factor of',us,'\n')        
cv2_imshow(down1)


[r,c] = np.shape(down)
us1 = 4
[r1,c1] = [int(np.ceil(r * us1)), int(np.ceil(c * us1))]
up1 = np.zeros((r1,c1), dtype=int)
for i in range(r1):
    for j in range(c1):
        x = int(np.floor(i / us1))
        y = int(np.floor(j / us1))
        up1[i, j] = down[x, y]
print('Shape1 is ',up1.shape)

[r,c] = np.shape(img)
us2 = 2
[r1,c1] = [int(np.ceil(r * us2)), int(np.ceil(c * us2))]
up2 = np.zeros((r1,c1), dtype=int)
for i in range(r1):
    for j in range(c1):
        x = int(np.floor(i / us2))
        y = int(np.floor(j / us2))
        up2[i, j] = img[x, y]
print('Shape2 is ',up2.shape)

plt.figure(figsize=(20,20))
plt.subplot(1,2,1)
plt.title('Restored original image by upsampling by 4')
plt.imshow(up1,cmap="gray")
plt.subplot(1,2,2)
plt.title('Original image upsampled by 2')
plt.imshow(up2,cmap='gray')

[m,n]=img.shape
print('Original image has a shape:' , (m,n))
b = int(input("Enter the number of bits for the new image: "))

img2= np.zeros((m,n), dtype= np.int) #inal 
levels= 2**b 
for i in range(m):
    for j in range(n):
        img2[i][j] = img[i][j]*(levels/256) 

f, axarr = plt.subplots(nrows=1,ncols=2)
plt.sca(axarr[0]); 
plt.imshow(img,cmap="gray"); plt.title('Original Image')
plt.sca(axarr[1]); 
plt.imshow(img2,cmap="gray"); plt.title('New image with bits = %i' %b)
plt.show()

for k in range(1,7): 
  levels = 2**k 
  imgnew= np.zeros((m,n), dtype= np.int) 
  for i in range(m):
    for j in range(n):
      imgnew[i][j] = img[i][j]*(levels/256) 
  plt.figure(figsize=(3,3))
  plt.title('Image with bits  =$ %i' %k)
  plt.imshow(imgnew,cmap='gray') 
     
