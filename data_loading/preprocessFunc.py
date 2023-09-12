import cv2
import numpy as np
from skimage.feature import canny
from skimage.filters import sobel
from skimage.transform import hough_line, hough_line_peaks

def apply_canny(image):
    canny_img = canny(image, 6)
    return sobel(canny_img)


def get_hough_lines(canny_img):
    h, theta, d = hough_line(canny_img)
    lines = list()
    print('\nAll hough lines')
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        print("Angle: {:.2f}, Dist: {:.2f}".format(np.degrees(angle), dist))
        x1 = 0
        y1 = (dist - x1 * np.cos(angle)) / np.sin(angle)
        x2 = canny_img.shape[1]
        y2 = (dist - x2 * np.cos(angle)) / np.sin(angle)
        lines.append({
            'dist': dist,
            'angle': np.degrees(angle),
            'point1': [x1, y1],
            'point2': [x2, y2]
        })
    
    return lines

def removeArtifacts(image):
        hh, ww = image.shape[:2]

        # # convert to grayscale
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # apply otsu thresholding
        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)[1] 

        # apply morphology close to remove small regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # apply morphology open to separate breast from other regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

        # apply morphology dilate to compensate for otsu threshold not getting some areas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35,35))
        morph = cv2.morphologyEx(morph, cv2.MORPH_DILATE, kernel)

        # get largest contour
        contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        big_contour = max(contours, key=cv2.contourArea)
        big_contour_area = cv2.contourArea(big_contour)

        # draw all contours but the largest as white filled on black background as mask
        mask = np.zeros((hh,ww), dtype=np.uint8)
        for cntr in contours:
            area = cv2.contourArea(cntr)
            if area != big_contour_area:
                cv2.drawContours(mask, [cntr], 0, 255, cv2.FILLED)
            
        # invert mask
        mask = 255 - mask

        # apply mask to image
        result = cv2.bitwise_and(image, image, mask=mask)
        return result
    
    