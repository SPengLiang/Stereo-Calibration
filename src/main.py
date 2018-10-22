#!usr/bin/env/ python
# _*_ coding:utf-8 _*_

import cv2 as cv
import numpy as np
import os

def calibrate_single(file_dir):
    # 标定所用图像
    pic_name = os.listdir(file_dir)

    # 由于棋盘为二维平面，设定世界坐标系在棋盘上，一个单位代表一个棋盘宽度，产生世界坐标系三维坐标
    real_cor = np.zeros((9 * 6, 3), np.float32)
    real_cor[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    real_points = []
    pic_points = []

    for pic in pic_name:
        pic_path = os.path.join(file_dir, pic)
        pic_data = cv.imread(pic_path)

        # 寻找到棋盘角点
        succ, pic_cor = cv.findChessboardCorners(pic_data, (9, 6), None)

        if succ:
            # 添加每幅图的对应3D-2D坐标
            pic_cor = pic_cor.reshape(-1, 2)
            pic_points.append(pic_cor)

            real_points.append(real_cor)

    _, cameraMatrix, distCoeffs, _, _ = cv.calibrateCamera(real_points, pic_points, (480, 640), None, None)
    return np.array(real_points), np.array(pic_points), cameraMatrix, distCoeffs


if __name__ == '__main__':
    #标定所用图片文件夹
    left_pic_dir = r'..\pic\left'
    right_pic_dir = r'..\pic\right'
    #单目标定
    real_points, left_pic_points, left_cameraMatrix, left_distCoeffs = calibrate_single(left_pic_dir)
    _, right_pic_points,  right_cameraMatrix, right_distCoeffs = calibrate_single(right_pic_dir)

    size = (640, 480)#图片尺寸

    #进行双目标定
    _, _, _, _, _, R, T, E, F = cv.stereoCalibrate(real_points, left_pic_points, right_pic_points,
                                                left_cameraMatrix, left_distCoeffs,
                                                right_cameraMatrix, right_distCoeffs, size)

    #相机坐标系转换
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv.stereoRectify(left_cameraMatrix, left_distCoeffs,
                                                                     right_cameraMatrix, right_distCoeffs,
                                                                     size, R, T)
    #减小畸变
    left_map1, leftmap2 = cv.initUndistortRectifyMap(left_cameraMatrix, left_distCoeffs, R1,
                                                     P1, size=size, m1type=cv.CV_16SC2)
    right_map1, right_map2 = cv.initUndistortRectifyMap(right_cameraMatrix, right_distCoeffs, R2,
                                                        P2, size=size, m1type=cv.CV_16SC2)
    #测试图片
    org = cv.imread(r'..\pic\left\left01.jpg')
    org2 = cv.imread(r'..\pic\right\right01.jpg')

    #显示线条，方便比较
    dst = cv.remap(org, left_map1, leftmap2, cv.INTER_LINEAR)
    for i in range(20):
        cv.line(dst, (0, i*24), (640, i*24), (0,255,0), 1)

    dst2 = cv.remap(org2, right_map1, right_map2, cv.INTER_LINEAR)
    for i in range(20):
        cv.line(dst2, (0, i*24), (640, i*24), (0,255,0), 1)

    canvas = np.zeros((480,1280,3), dtype="uint8")
    canvas[:, :640] = dst
    canvas[:, 640:] = dst2
    cv.line(canvas, (640, 0), (640, 480), (255,0,0), 1)
    cv.imshow('canvas', canvas)
    cv.waitKey(0)
    cv.destroyAllWindows()

    print("left param: ", left_cameraMatrix, left_distCoeffs)
    print("right param: ", right_cameraMatrix, right_distCoeffs)
    print("stereo param: ", R, T, E, F)