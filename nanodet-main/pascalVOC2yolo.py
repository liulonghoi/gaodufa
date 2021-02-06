# import os
# import random
# trainval_percent = 0.1
# train_percent = 0.9
# xmlfilepath = '/home/swjtu/darknet/VOCdevkit/VOC2007/Annotations'
# txtsavepath = '/home/swjtu/darknet/VOCdevkit/VOC2007/ImageSets'
# total_xml = os.listdir(xmlfilepath)
# num = len(total_xml)
# list = range(num)
# tv = int(num * trainval_percent)
# tr = int(tv * train_percent)
# trainval = random.sample(list, tv)
# train = random.sample(trainval, tr)
# ftrainval = open('/home/swjtu/darknet/VOCdevkit/VOC2007/ImageSets/trainval.txt', 'w')
# ftest = open('/home/swjtu/darknet/VOCdevkit/VOC2007/ImageSets/test.txt', 'w')
# ftrain = open('/home/swjtu/darknet/VOCdevkit/VOC2007/ImageSets/train.txt', 'w')
# fval = open('/home/swjtu/darknet/VOCdevkit/VOC2007/ImageSets/val.txt', 'w')
# for i in list:
#     name = total_xml[i][:-4] + '\n'
#     if i in trainval:
#         ftrainval.write(name)
#         if i in train:
#             ftest.write(name)
#         else:
#             fval.write(name)
#     else:
#         ftrain.write(name)
# ftrainval.close()
# ftrain.close()
# fval.close()
# ftest.close()
#




import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
sets = ['train', 'test','val']
classes = ['0-hand','1-hand','2-hand','3-hand','4-hand','5-hand']
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)
def convert_annotation(image_id):
    in_file = open('/home/swjtu/darknet/VOCdevkit/VOC2007/Annotations/%s.xml' % (image_id))
    out_file = open('/home/swjtu/darknet/VOCdevkit/VOC2007/labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
wd = getcwd()
print(wd)
for image_set in sets:
    if not os.path.exists('/home/swjtu/darknet/VOCdevkit/VOC2007/labels/'):
        os.makedirs('/home/swjtu/darknet/VOCdevkit/VOC2007/labels/')
    image_ids = open('/home/swjtu/darknet/VOCdevkit/VOC2007/ImageSets/%s.txt' % (image_set)).read().strip().split()
    list_file = open('/home/swjtu/darknet/VOCdevkit/VOC2007/%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write('/home/swjtu/darknet/VOCdevkit/VOC2007/images/%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()