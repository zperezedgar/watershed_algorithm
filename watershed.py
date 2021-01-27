import numpy as np
import cv2

def watershed(img):
    """
    This function implements image segmentation to img by using the watershed algorithm.
    :param:
    img: The image in RGB format to which the algorithm will be applied.
    :return:
    The image obtained with the watershed algorithm. 
    In this image the sure background of img has been identified and colored black.
    The rest of the image remains the same.
    """
    
    img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #demo
    #gray = hsv[:,:,1] + hsv[:,:,2]# demo

    ret, thresh = cv2.threshold(gray,0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # remove noise
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 1)
    #opening = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 1)

    # Find the sure background region
    sure_bg = cv2.dilate(opening, kernel, iterations=8)

    # Find the sure foreground region.
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    # Find the unkown region.
    unkown = cv2.subtract(sure_bg, sure_fg)

    # Label the foreground objects.
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1.
    markers += 1

    # Label the unkown region as 0.
    markers[unkown==255] = 0

    markers = cv2.watershed(img, markers)
    img[markers==1] = [0,0,0] #sure background
    #img[markers==-1] = [0,0,0] #boundary region
    
    return img
