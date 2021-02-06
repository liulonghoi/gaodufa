# coding=utf-8
import os


path = '/home/user/llk/nanodet-main/voc12ps/anno'  # 对该路径下所有文件重命名排序
count = 2343
file_list = os.listdir(path)  # 该文件夹下所有的文件（包括文件夹）

for file in file_list:  # 遍历所有文件
    olddir=os.path.join(path, file)  # 原来的文件路径
    if os.path.isdir(olddir):  # 如果是文件夹则跳过
        continue
    filename=os.path.splitext(file)[0]  # 文件名
    filetype=os.path.splitext(file)[1]  # 文件扩展名
    newdir=os.path.join(path, str(count) + filetype)  # 新的文件路径
    os.rename(olddir, newdir)  # 重命名
    count += 1
# import xml.etree.ElementTree as ET
#
# import os
#
#
# # 批量修改整个文件夹所有的xml文件
#
# def change_all_xml(xml_path):
#     # filelist = os.listdir(xml_path)
#     filelist = [os.path.join(xml_path,filename)for filename in os.listdir(xml_path) if filename.endswith('.xml')]
#
#
#     # 打开xml文档
#
#     for xmlfile in filelist:
#         print(xmlfile)
#         doc = ET.parse(xmlfile)
#
#         root = doc.getroot()
#
#         sub1 = root.find('path')  # 找到filename标签，
#         print(sub1.text)
#         old= str(sub1.text)
#
#         print(old)
#         old1 = old.split("\\")[-1]
#         sub1.text = "/home/user/llk/Matlab_object_detection/boxcover/"+old1 # 修改标签内容
#
#         doc.write(xmlfile)  # 保存修改
#
# change_all_xml(r'/home/user/llk/Matlab_object_detection/boxcover')