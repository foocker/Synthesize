# https://www.codetd.com/article/5592888

# Python 2/3 compatibility
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

import cv2 as cv

from contextlib import contextmanager


def clock():
    return cv.getTickCount() / cv.getTickFrequency()

@contextmanager
def Timer(msg):
    print(msg, '...',)
    start = clock()
    try:
        yield
    finally:
        print("%.2f ms" % ((clock()-start)*1000))