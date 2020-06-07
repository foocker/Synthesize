import numpy as np
import cv2

from pyzbar.pyzbar import decode


# https://www.voidking.com/dev-gp-image-tilt-correction/

def is_exist_barcode():
    pass

def rotate_bound(image, angle):

    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def bar_box_theta(image):
    # load the image and convert it to grayscale 
    # image = cv2.imread(file)
    # h, w, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # compute the Scharr gradient magnitude reprtesentation of the images 
    # in both the x and y direction 
    gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
    gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)

    #substract the y-gradient from the x-gradient 
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    #blur and threshold the image 
    blurred = cv2.blur(gradient, (9,9))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)   # can be auto thresh

    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # perform a series of erosions and dilations 
    closed = cv2.erode(closed, None, iterations = 1)    # when cig outer is white...
    closed = cv2.dilate(closed, None, iterations = 1)

    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    ###### surpose there is one barcode in the image ########
    cv2.imshow('closed', closed)

    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # may 3 return
    # print(cnts, 'kkkk')    # empty and check barcode
    c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]    # may empty
    
    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)    # boxPoints: 左上角(x,y)，（width, height）,旋转角度
    # box = np.int0(cv2.boxPoints(rect))
    # cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    # cv2.imshow('box', image)
    # cv2.waitKey()
    theta = rect[2]
    # if theta < -45:
    #     theta_reverse = -(90+ theta)
    # else:
    #     theta_reverse = -theta
    # theta_reverse = theta
    # print('theta is', theta, theta_reverse)
    return theta


def bar_box_theta_2(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # equalize lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # edge enhancement
    edge_enh = cv2.Laplacian(gray, ddepth = cv2.CV_8U, 
                             ksize = 3, scale = 1, delta = 0)
    # cv2.imshow("Edges", edge_enh)
    # retval = cv2.imwrite("edge_enh.jpg", edge_enh)

    # bilateral blur, which keeps edges
    blurred = cv2.bilateralFilter(edge_enh, 13, 50, 50)

    # use simple thresholding. adaptive thresholding might be more robust
    (_, thresh) = cv2.threshold(blurred, 55, 255, cv2.THRESH_BINARY)
    # cv2.imshow("Thresholded", thresh)
    # retval = cv2.imwrite("thresh.jpg", thresh)

    # do some morphology to isolate just the barcode blob
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 4)
    # cv2.imshow("After morphology", closed)
    
    # retval = cv2.imwrite("closed.jpg", closed)

    # find contours left in the image
    (_,cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
    rect = cv2.minAreaRect(c)
    # box = np.int0(cv2.boxPoints(rect))
    # cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    # cv2.imshow("found barcode", image)
    # retval = cv2.imwrite("found.jpg", image)
    # cv2.waitKey()
    return rect[2]



def barbox_rotatedcrop(orgimg, barboxPoints):
    pass


def barcode_number(img):
    barcodes = decode(img)
    if len(barcodes):
        bc = barcodes[0]
        return {'CodesType':bc.type, 'codenumber':bc.data.decode('gbk')}
    return 


def generate_rotated_img(file):
    img = cv2.imread(file)
    angle = np.random.randint(-90, 0)
    rotatde_img = rotate_bound(img, angle)
    cv2.imshow('rotated', rotatde_img)
    cv2.waitKey()
    cv2.imwrite(r'images/rotated_{}_{}'.format(angle, file.split('/')[-1]),rotatde_img)


def barcode_rotated(file):
    img = cv2.imread(file)
    barinfo = barcode_number(img)
    if barinfo is None:
        # theta_reverse = bar_box_theta(img)
        theta_reverse = bar_box_theta_2(img)
        rotated_img = rotate_bound(img, theta_reverse)
        barinfo = barcode_number(rotated_img)
        # print(barinfo)
        # cv2.imshow("roated img", rotated_img)
        # cv2.waitKey()

    return barinfo



# barinfo = barcode_rotated('')    # pyzbar is not good...
# print(barinfo)

        