#import libraries
import cv2
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

#to open image using OpenCV and matplotlib

img1= cv2.imread("lake.tif")
print('Show using matplotlib\n')
plt.imshow(img1)

#to open image using OpenCV and writing the image

print('Show using OpenCV\n')
cv2_imshow(img1)
cv2.imwrite('lake1.tif', img1)

#to determine the image type
type(img1)

#to find image shape, dimension and size
print('Shape: ',img1.shape) #returns a tuple (R x C x channels)
print('Dimensions: ',img1.ndim) #returns dimension of the array
print('Number of pixels: ',img1.size) #returns number of elements/pixels in the image

#to open a colour image as greyscale using OpenCV
img2=cv2.imread("/content/lena_color_256.tif")
img3= cv2.imread("/content/lena_color_256.tif",0)
print('Image with colour\n')
cv2_imshow(img2)
print('\n')
print('Image without colour\n')
cv2_imshow(img3)
