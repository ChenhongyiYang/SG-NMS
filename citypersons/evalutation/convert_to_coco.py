import json
import os
import numpy as np


gt_json = '/Users/yangchenhongyi/Documents/citypersons/shanshanzhang-citypersons-ae6814faa761/evaluation/val_gt.json'

def get_image_id(val_gt):
    anns = json.load(open(val_gt))
    id_dict = {}
    for item in anns['images']:
        id_dict[item['im_name'].split('.')[0]] = item['id']
    return id_dict

def get_image_name(val_gt):
    anns = json.load(open(val_gt))
    name_dict = {}
    for item in anns['images']:
        name_dict[item['id']] = item['im_name'].split('.')[0]
    return name_dict


def read_pfile(pfile):
    lines = open(pfile, 'r').readlines()
    rois = []
    scores = []
    for line in lines:
        line = line.strip().split(' ')
        ymin, xmin, ymax, xmax = [int(c) for c in line[1:5]]
        scores.append(float(line[5]))
        rois.append([xmin,ymin,xmax-xmin,ymax-ymin])
    return rois, scores

def write_gt_as_result():
    anns = json.load(open(gt_json))
    id_dict = get_image_id(gt_json)

    results = []
    for item in anns['annotations']:
        results.append({'image_id':item['image_id'],
                        'category_id':1,
                        'bbox':item['bbox'],
                        'score':1.})
    with open('/Users/yangchenhongyi/Documents/citypersons/shanshanzhang-citypersons-ae6814faa761/evaluation/val_gt_det.json', 'w') as outfile:
        json.dump(results, outfile)

def convert_to_coco(txt_dir, val_gt, out_json):
    file_list = os.listdir(txt_dir)
    id_dict = get_image_id(val_gt)

    results = []

    for txt in file_list:
        if txt.endswith('txt'):
            boxes, scores = read_pfile(os.path.join(txt_dir,txt))
            for i in range(len(scores)):
                results.append({'image_id': id_dict[txt.split('.')[0]],
                                'category_id': 1,
                                'bbox': boxes[i],
                                'score': scores[i]})
    with open(out_json, 'w') as outfile:
        json.dump(results, outfile)


def convert_to_test_format(txt_dir, out_json):
    file_list = os.listdir(txt_dir)
    file_list.sort()

    results = []

    for k, txt in enumerate(file_list):
        if not txt.endswith('txt'):
            continue
        boxes, scores = read_pfile(os.path.join(txt_dir, txt))
        for i in range(len(scores)):
            results.append({'image_id': k+1,
                            'category_id': 1,
                            'bbox': boxes[i],
                            'score': scores[i]})
    with open(out_json, 'w') as outfile:
        json.dump(results, outfile)






if __name__ == '__main__':
    txt_dir = '/Users/yangchenhongyi/Documents/TEMP/cp_result'
    val_gt = gt_json
    out_json = '/Users/yangchenhongyi/Documents/citypersons/shanshanzhang-citypersons-ae6814faa761/evaluation/test_det.json'
    convert_to_coco(txt_dir, val_gt, out_json)

























