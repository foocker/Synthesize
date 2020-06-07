import cv2

global img
global point1, point2


def on_mouse(event, x, y, flags, param):
    global img, point1, point2
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:         #左键点击
        point1 = (x,y)
        cv2.circle(img2, point1, 10, (0,255,0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):   #按住左键拖曳
        cv2.rectangle(img2, point1, (x,y), (255,0,0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:         #左键释放
        point2 = (x,y)
        cv2.rectangle(img2, point1, point2, (0,0,255), 5) 
        cv2.imshow('image', img2)
        min_x = min(point1[0], point2[0])     
        min_y = min(point1[1], point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] -point2[1])
        cut_img = img[min_y:min_y+height, min_x:min_x+width]
        cv2.imwrite('TS_KL2H_1149_0.png', cut_img)

def main():
    global img
    img = cv2.imread('')
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse)
    cv2.imshow('image', img)
    cv2.waitKey(0)


"""
鼠标响应参数:
    event:
        define CV_EVENT_MOUSEMOVE 0 滑动
        define CV_EVENT_LBUTTONDOWN 1 击键点击
        define CV_EVENT_RBUTTONDOWN 2 右键点击
        define CV_EVENT_MBUTTONDOWN 3 中间点击
        define CV_EVENT_LBUTTONUP 4 左键释放
        define CV_EVENT_RBUTTONUP 5 右键释放
        define CV_EVENT_MBUTTONUP 6 中间释放
        define CV_EVENT_LBUTTONDBLCLK 7 左键双击
        define CV_EVENT_RBUTTONDBLCLK 8 右键双击
        define CV_EVENT_MBUTTONDBLCLK 9 中间释放
    flags:
        define CV_EVENT_FLAG_LBUTTON 1 左键拖拽
        define CV_EVENT_FLAG_RBUTTON 2 右键拖拽
        define CV_EVENT_FLAG_MBUTTON 4 中间拖拽
        define CV_EVENT_FLAG_CTRLKEY 8 (8~15)按Ctrl不放事件
        define CV_EVENT_FLAG_SHIFTKEY 16 (16~31)按Shift不放事件
        define CV_EVENT_FLAG_ALTKEY 32 (32~39)按Alt不放事件
"""

if __name__ == '__main__':
    main()
