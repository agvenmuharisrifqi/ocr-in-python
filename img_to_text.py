import cv2
import pytesseract
import numpy as np

# Read IMAGE default color
img = cv2.imread('imgText.png')

# Edit Image to Gray
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Edit Image to Black and White
black, white = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# black, white = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)

# To Invers Color Image
img_invert = cv2.bitwise_not(white)

kernel = np.ones((1, 1), np.uint8)

img_erosion = cv2.erode(img_invert, kernel, iterations=1)
img_dilation = cv2.dilate(img_invert, kernel, iterations=1)
img_morph_opening = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel, iterations=1)
img_morph_closing = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel, iterations=1)
img_morph_gradient = cv2.morphologyEx(white, cv2.MORPH_GRADIENT, kernel, iterations=1)
img_morph_top_hat = cv2.morphologyEx(white, cv2.MORPH_TOPHAT, kernel, iterations=1)
img_morph_black_hat = cv2.morphologyEx(white, cv2.MORPH_BLACKHAT, kernel, iterations=1)

# Get Text From Image
text_result = pytesseract.image_to_string(img_invert)
# text = pytesseract.image_to_alto_xml(img_black)
print(text_result[:-1])

# Create WINDOW to display IMAGE
cv2.imshow('Erosion', img_erosion)
cv2.imshow('Dilation', img_dilation)
cv2.imshow('Opening', img_morph_opening)
cv2.imshow('Closing', img_morph_closing)
cv2.imshow('Gradient', img_morph_gradient)
cv2.imshow('Top Hat', img_morph_top_hat)
cv2.imshow('Black Hat', img_morph_black_hat)
   
# De-allocate any associated memory usage 
if cv2.waitKey(5000):
    cv2.destroyAllWindows()


