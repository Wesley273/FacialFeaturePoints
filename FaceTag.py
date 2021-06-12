import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    r'sources\shape_predictor_68_face_landmarks.dat')
cap = cv2.VideoCapture(0)  # 设置摄像头 0是默认的摄像头 如果有多个摄像头的话，可以设置1,2,3....
while True:  # 进入无限循环
    # cv2读取图像
    ret, img = cap.read()
    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rects = detector(img_gray, 0)  #返回人脸矩形框四点坐标，多个人脸则有多组坐标
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
        for idx, point in enumerate(landmarks):
            # 68点的坐标
            pos = (point[0, 0], point[0, 1])
            # 利用cv2.circle给每个特征点画一个圈，共68个
            cv2.circle(img, pos, 5, color=(0, 255, 0))
    cv2.imshow('frame', img)  # 将img的值显示出来 有两个参数 前一个是窗口名字，后面是值
    c = cv2.waitKey(1)  # 判断退出的条件 当按下'Q'键的时候呢，就退出
    if c == ord('q'):
        break
cap.release()  # 常规操作
cv2.destroyAllWindows()
