import cv2

img=cv2.imread('test1.jpg')

# Prints Dimensions of the image
print(img.shape)

# Display the image
cv2.imshow("original", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cropped_image = img[80:280, 150:330] # Slicing to crop the image

# Display the cropped image
cv2.imshow("cropped", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
