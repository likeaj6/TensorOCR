import cv2
import numpy as np

def smoothImage(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    ret, threshold = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # filtered = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)

    return threshold


def cleanImage(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    filtered = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)

    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = smoothImage(img)
    or_img = cv2.bitwise_or(img, closing)
    return or_img

i = 0
image = cv2.imread("cca.jpg")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # grayscale
_,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV) # threshold
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
dilated = cv2.dilate(thresh,kernel,iterations = 13) # dilate
_, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours

# for each contour found, draw a rectangle around it on original image
for contour in contours:
    # get rectangle bounding contour
    [x,y,w,h] = cv2.boundingRect(contour)

    # discard areas that are too large
    if h>300 and w>300:
        continue

    # discard areas that are too small
    if h<40 or w<40:
        continue

    # draw rectangle around contour on original image
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)
    region = image[y: y + h, x: x + w]
# write original image with added contours to disk

    region = cleanImage(region)
    cv2.imwrite("test"+str(i)+".jpg",region)
    i = i+1
    # while True:
    #     k = cv2.waitKey(5)
    #     if k % 0x100 == 27:
    #         break
