# https://github.com/pablospe/MVCDemo
# http://blog.csdn.net/hjimce/article/details/45716603
# https://github.com/Erkaman/poisson_blend
# Standard imports
import cv2
import numpy as np 

# Read images
src = cv2.imread("./test_images/airplane.jpg")
dst = cv2.imread("./test_images/sky.jpg")

# Create a rough mask around the airplane.
src_mask = np.zeros(src.shape, src.dtype)
poly = np.array([ [4,80], [30,54], [151,63], [254,37], [298,90], [272,134], [43,122] ], np.int32)
cv2.fillPoly(src_mask, [poly], (255, 255, 255))

# This is where the CENTER of the airplane will be placed
center = (800,100)

# Clone seamlessly.
output = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)

# Save result
cv2.imwrite("./test_images/opencv-seamless-cloning-example.jpg", output)


# if __name__ == "__main__":
#     pass