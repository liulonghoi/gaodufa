#！/usr/bin/python3.7

import cv2
import os
import time
import torch
import argparse
from nanodet.util import cfg, load_config, Logger
from nanodet.model.arch import build_model
from nanodet.util import load_model_weight
from nanodet.data.transform import Pipeline
from nanodet.util import distance2bbox, bbox2distance, overlay_bbox_cv
import numpy as np
image_ext = ['.jpg', '.jpeg', '.webp', '.bmp', '.png']
video_ext = ['mp4', 'mov', 'avi', 'mkv']



def parse_args():
    parser = argparse.ArgumentParser()
    # 修改以下配置参数路径
    parser.add_argument('--config', default='D:\Desktop\高度阀优化v7/nanodet_v7.yml',help='model config file path')
    parser.add_argument('--model', default ='D:\Desktop\高度阀优化v7/model_v7.pth',help='model file path')
    parser.add_argument('--test_path', default='D:/Desktop/jiaoda_test2', help='path to test images or video')
    parser.add_argument('--save_path', default='../save_dir', help='path to save images or video')
    args = parser.parse_args()
    return args


class Predictor(object):
    def __init__(self, cfg, model_path, logger, device='cuda:0'):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
    def inference(self, img):
        img_info = {}
        if isinstance(img, str):
            img_info['file_name'] = os.path.basename(img)
            # img = cv2.imdecode(np.fromfile(img, dtype=np.uint8), -1)   # 若需要检测的样本含中文或路径中含有中文可使用cv2.imdecode读取图片，注释掉下一行
            img = cv2.imread(img)
        else:
            img_info['file_name'] = None
        # image preprocessing
        img = img[450:img.shape[0], 0:img.shape[1]]  # 裁剪坐标为[y0:y1, x0:x1]
        img = img * float(3) + 10
        img[img > 255] = 255
        img = np.round(img)
        img = img.astype(np.uint8)
        height, width = img.shape[:2]
        img_info['height'] = height
        img_info['width'] = width
        meta = dict(img_info=img_info,
                    raw_img=img,
                    img=img)
        meta = self.pipeline(meta, self.cfg.data.val.input_size)
        meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return image_names
def save_img(img,save_path,nake_name,dets, class_names, score_thres):
    result = overlay_bbox_cv(img, dets, class_names, score_thresh=score_thres)
    cv2.imwrite(save_path+'/'+nake_name,result)
    # cv2.imencode('.jpg', result)[1].tofile(save_path + '/' +nake_name) # 若需要检测的样本含中文或路径中含有中文可使用cv2.imdecode读取图片，注释掉下一行

def count_num(dets,score_thresh):
    """

    Args:
        dets: nanodet返回的结果
        score_thresh: 对不同的待检测部件设置的置信度最低阈值

    Returns:
            返回布尔值，即是否为正样本
    """

    all_box = []    # 预定义一个空列表用于保存各个零部件的检测信息。包括label,anchor bboxes,以及score
    for label in dets: # 遍历dets字典，提取不同的label的检出框信息
        for bbox in dets[label]:
            score = bbox[-1]
            if score>score_thresh: # 判断各个检出框的置信度是否小于最低阈值
                x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                all_box.append([label, x0, y0, x1, y1, score])
    if len(all_box) == 6:
        np_all_box = np.array(all_box)
        score = np_all_box[:,5]
        np_label = np_all_box[:,0]
        bolt_np_label = (np_label==0)
        bolt_score = (score[bolt_np_label])  # 提取出bolt的置信度
        bolt_num = sum(bolt_score>0.7) # 判断bolt的置信度是否大于0.7
        nut2_np_label = (np_label==2)
        nut2_score = (score[nut2_np_label]) # 提取出nut2的置信度
        nut2_num = sum(nut2_score>0.7) # 判断nut2的置信度是否大于0.7
        if (bolt_num==2)and(nut2_num==1):
            return True
        else:
            return False
    else:
        return False

def main():
    args = parse_args()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    load_config(cfg, args.config)
    logger = Logger(-1, use_tensorboard=False)
    predictor = Predictor(cfg, args.model, logger, device='cpu')
    logger.log('Press "Esc", "q" or "Q" to exit.')
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(os.path.join(args.save_path, 'neg')):
        os.makedirs(os.path.join(args.save_path, 'neg'))
    if not os.path.exists(os.path.join(args.save_path, 'posi')):
        os.makedirs(os.path.join(args.save_path, 'posi'))
    if os.path.isdir(args.test_path):
        files = get_image_list(args.test_path)
    else:
        files = [args.test_path]
    files.sort()
    count = 0
    for image_name in files:
        meta, res = predictor.inference(image_name)
        nake_name = image_name.split('\\')[-1]
        if not (count_num(res,0.6)):
            print("%s is a negative sample!" % nake_name)
            save_img(meta['raw_img'], args.save_path + '/neg', nake_name, res, cfg.class_names, 0.6)
            count += 1
        else:

            save_img(meta['raw_img'], args.save_path + '/posi', nake_name, res, cfg.class_names, 0.6)
    print("the num of negative sample is %d" % count)

if __name__ == '__main__':
    main()
