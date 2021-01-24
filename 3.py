##Plot Histogram of an image without using inbuilt function


#import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# load required image as a greyscale image (0 in imread function converts any rgb image to greyscale)
img = cv2.imread('woman_darkhair.tif',0) 
                   
print('Directly visualizing the histogram using a matplotlib function')
plt.hist(img.ravel(),256,[0,256]); plt.show()

def hist_plot(img):                             
  row,col= img.shape              # determine shape of the image by assigning number of rows to m and number of columns to n 
  count=[]                        # define an empty list to store number of pixels of each intensity value
  r= []                           # define an empty list to store all the intensity values
  for k in range(0,256):          # use for loop to traverse each intensity value
      r.append(k)
      countint=0                    # initialise the counter for the current pixel value according to k
      for i in range(row):          # for loop to traverse each pixel in the image along the rows
          for j in range(col):      # for loop to traverse each pixel in the image along the cloumns
              if img[i,j]==k:       # fulfilling the condition of intensity value of a particular pixel being equal to the instantaneous intensity value in k
                  countint+=1       # increment the counter by 1 at the end of each complete iteration
      count.append(countint)        # append new count value to original list
  
  plt.figure(figsize=(15,15))       # define figure size
  plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=1) # add spaces between subplots
  plt.subplot(2,2,1)
  plt.imshow(img,cmap='gray',vmin = 0, vmax = 255) #the vmax and vmin arguments prevetn matplotlib from auto-enhancing the images
  plt.title('Original image')      #display original image
  plt.subplot(2,2,2)        
  plt.stem(r,count)                #plotting the histogram discretely
  plt.xlabel('Intensity values')    
  plt.ylabel('No of pixels')
  plt.title('Discrete histogram plot')   
#hist_plot(img)

"""## Plot histogram of different images and classify them as low contrast, high contrast, dark and bright images."""

imga = cv2.imread('320a.tif',0)
hist_plot(imga)

imgb = cv2.imread('320b.tif',0)
hist_plot(imgb)

imgc = cv2.imread('320c.tif',0)
hist_plot(imgc)

imgd = cv2.imread('320d.tif',0)
hist_plot(imgd)

"""##Histogram stretching"""

def str_hist_plot(img):                             
  row,col= img.shape              # determine shape of the image by assigning number of rows to m and number of columns to n 
  count=[]                        # define an empty list to store number of pixels of each intensity value
  r= []                           # define an empty list to store all the intensity values
  for k in range(0,256):          # use for loop to traverse each intensity value
      r.append(k)
      countint=0                    # initialise the counter for the current pixel value according to k
      for i in range(row):          # for loop to traverse each pixel in the image along the rows
          for j in range(col):      # for loop to traverse each pixel in the image along the cloumns
              if img[i,j]==k:       # fulfilling the condition of intensity value of a particular pixel being equal to the instantaneous intensity value in k
                  countint+=1       # increment the counter by 1 at the end of each complete iteration
      count.append(countint)        # append new count value to original list
  
  plt.figure(figsize=(15,15))
  plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=1)
  plt.subplot(2,2,1)
  plt.imshow(img,cmap='gray',vmin = 0, vmax = 255)
  plt.title('Stretched image')      #display original image
  plt.subplot(2,2,2)        
  plt.stem(r,count)                #plotting the histogram discretely
  plt.xlabel('Intensity values')    
  plt.ylabel('No of pixels')
  plt.title('Discrete histogram plot')  

def hist_stretched(img):
  rmin = img.min()
  rmax = img.max()
  smin = 0
  smax = 255
  row , col = img.shape
  c = ((smax-smin)/(rmax-rmin)) #transformation function to obtain stretching
  stretch = np.zeros((row,col),dtype=np.int)
  for i in range(row):
    for j in range(col):
      stretch[i,j] = ((img[i,j] - rmin)*c) + smin

imga = cv2.imread('320a.tif',0)
hist_stretched(imga)

imgb = cv2.imread('320b.tif',0)
hist_stretched(imgb)

imgc = cv2.imread('320c.tif',0)
hist_stretched(imgc)

imgd = cv2.imread('320d.tif',0)
hist_stretched(imgd)

"""##Histogram equalization

"""

#histogram equalization

#import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# load required image as a greyscale image (0 in imread function converts any rgb image to greyscale)
img = cv2.imread('Fig0316(2)(2nd_from_top).tif',0) 
# convert image into a numpy array
img = np.asarray(img)

# put pixels in a 1D array by flattening out the array
flat = img.flatten()

# plot the histogram
plt.hist(flat, bins=255)
plt.xlabel('Intensity values')    
plt.ylabel('No of pixels')
plt.title('Discrete histogram plot')
plt.show()

#define an alternate function to plot the histogram; previous function may also be used
def get_histogram(image, bins):
    # create an array with size of bins and set to zeros
    histogram = np.zeros(bins) #bins define the total number of groups of equal width that form the histogrm; here bins = 256 because we want a range from 0 to 255
    
    # loop through pixels and increment count of pixels
    for pixel in image:
        histogram[pixel] += 1
    
    return histogram

hist = get_histogram(img, 256)

#define cumulative sum function
def cumulativesum(a):
    a = iter(a) 
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)

cs = cumulativesum(hist)

plt.plot(cs)

new = cs[img]

new = np.reshape(new, img.shape)

# set up side-by-side image display
plt.figure(figsize=(15,15))
plt.subplot(2,2,1)
plt.imshow(img, cmap='gray', vmax=255,vmin=0)

# display the new image
plt.subplot(2,2,2)
plt.imshow(new, cmap='gray')

img1= np.asarray(new)

# put pixels in a 1D array by flattening out the array
flat = img1.flatten()

# plot the histogram
plt.hist(flat, bins=255)

"""#Histogram equalization without inbuilt fn

"""

#histogram equalization

#import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

def imhist(im):
  # calculate the normalized histogram of an image
  m, n = im.shape
  h = [0] * 256 
  for i in range(m): #traverse row wise
    for j in range(n): #traverse column wise
      h[im[i, j]]+=1 #get array of number of pixel for each intensity value  
  return np.array(h)/(m*n) #get probability array

def cumsum(h):
	# find cumulative sum 
	return [sum(h[:i+1]) for i in range(len(h))]

def histeq(im):
	#calculate Histogram
  h = imhist(im)#find histogram parameters for original image
  cdf = np.array(cumsum(h)) #cumulative distribution function
  sk = np.round(255 * cdf) #finding transfer function values by performing rounding operation
  m,n = im.shape
  Y = np.zeros((m,n),dtype=np.int) #define new array of zeros equal in size to the original image
	# applying transfered values for each pixel
  for i in range(m): #traverse row wise
    for j in range(n): #traverse column wise
      Y[i, j] = sk[im[i, j]] #apply transfer function to original image to get the new image
  H = imhist(Y) #find histogram parameters for equalised image
	#return transformed image, original and new histogram, 
	# and transform function
  return Y , h, H, sk

img = cv2.imread('woman_blonde.tif',0) #load image as grayscale image

new_img, h, new_h, sk = histeq(img) #apply histogram equalizaition on the loaded image

# show old and new image for side-by-side comparison

# show original image
plt.figure(figsize=(15,15))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.subplot(2,3,1)
plt.imshow(img, cmap ='gray',vmin = 0, vmax = 255)
plt.title('Original image')
# show new image image
plt.subplot(2,3,2)
plt.imshow(new_img, cmap ='gray',vmin = 0, vmax = 255)
plt.title('Equalized image')
equal = cv2.equalizeHist(img)
plt.subplot(2,3,3)
plt.imshow(equal, cmap ='gray',vmin = 0, vmax = 255)
plt.title('Equalized image from OpenCV implementation')
plt.show()

# plot histograms and transfer function
fig = plt.figure(figsize=(15,15))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.25)
fig.add_subplot(2,2,1)
plt.plot(h) # plot original histogram
plt.xlabel('Intensity values')    
plt.ylabel('No of pixels')
plt.title('Original histogram') 

fig.add_subplot(2,2,2)
plt.plot(new_h) # plot hist of equalized image
plt.xlabel('Intensity values')    
plt.ylabel('No of pixels')
plt.title('Equalized histogram') 

fig.add_subplot(2,2,3)
plt.plot(sk) # plot transfer function
plt.xlabel('rk')    
plt.ylabel('sk')
plt.title('Transfer function') 

plt.show()
