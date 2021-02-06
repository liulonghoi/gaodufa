'''预处理图像文件（先裁剪图像为固定尺寸，再提高图像对比度）'''
import os
import cv2
import numpy as np
# 定义待批量预处理图像的路径地址
IMAGE_INPUT_PATH = '/home/user/llk/nanodet-main/test'
# 定义批量预处理后的图像存放地址
IMAGE_OUTPUT_PATH = '/home/user/llk/nanodet-main/test1'
a = 3
b = 10
for each_image in os.listdir(IMAGE_INPUT_PATH):
    image_input_fullname = IMAGE_INPUT_PATH + '/' + each_image
    img = cv2.imread(image_input_fullname)
    cropped = img[450:img.shape[0], 0:img.shape[1]]  # 裁剪坐标为[y0:y1, x0:x1]
    cropped = cropped * float(a) + b
    cropped[cropped > 255] = 255
    cropped = np.round(cropped)
    cropped =cropped.astype(np.uint8)

    image_output_fullname = IMAGE_OUTPUT_PATH + "/" + each_image
    cv2.imwrite(image_output_fullname,cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    print('{0} preprocessing done.'.format(each_image))
