import cv2
import numpy as np
from matplotlib import pyplot as plt
image =cv2.imread('he.png')
cv2.imshow('before histo',image)
image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
hist=cv2.calcHist([image],[0],None,[256],[0,256])
plt.plot(hist)
plt.figure()
image_hist=cv2.equalizeHist(image)
hist=cv2.calcHist([image_hist],[0],None,[256],[0,256])
plt.plot(hist)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(1, 2, 2), plt.imshow(image_hist, cmap='gray'), plt.title('Equalized Image')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()