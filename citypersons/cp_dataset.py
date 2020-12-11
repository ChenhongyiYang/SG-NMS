import os

import json
import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET

from preprocessing import tf_image



cp_width = 2048
cp_height = 1024

max_bbox_num = 100

def read_label(label_file, norm):
    tree = ET.parse(label_file)
    root = tree.getroot()

    classes = []
    bboxes = []

    for child in root:
        if child.tag == 'object':
            for sub_child in child:
                if sub_child.tag == 'name':
                    if sub_child.text == 'ped':
                        classes.append(1)
                    else:
                        classes.append(-1)
                if sub_child.tag == 'bndbox':
                    xmin, ymin, xmax, ymax = 0., 0., 0., 0.
                    for sub_sub_child in sub_child:
                        if sub_sub_child.tag == 'xmin':
                            xmin = float(sub_sub_child.text)
                        elif sub_sub_child.tag == 'ymin':
                            ymin = float(sub_sub_child.text)
                        elif sub_sub_child.tag == 'xmax':
                            xmax= float(sub_sub_child.text)
                        else:
                            ymax = float(sub_sub_child.text)
                    if norm:
                        xmin = xmin / cp_width
                        ymin = ymin / cp_height
                        xmax = xmax / cp_width
                        ymax = ymax / cp_height
                    bboxes.append([ymin,xmin,ymax,xmax])
    return classes, bboxes



def dataset_gener_with_batch(image_list, bbox_list, label_list,batch_size):
    def _parse_func(filename):
        img_string = tf.read_file(filename)
        img = tf.image.decode_png(img_string)
        img = tf_image.resize_image(img, (cp_height,cp_width),
                                          method=tf.image.ResizeMethod.BILINEAR,
                                          align_corners=False)
        img = tf.cast(img, dtype=tf.float32)
        img = tf.div(img, 255.)
        return img

    data_img = tf.data.Dataset.from_tensor_slices(image_list)
    data_bbox = tf.data.Dataset.from_tensor_slices(bbox_list)
    data_label = tf.data.Dataset.from_tensor_slices(label_list)

    data_img = data_img.map(_parse_func)
    dataset = tf.data.Dataset.zip((data_img, data_label, data_bbox)).shuffle(100).repeat().batch(batch_size)
    return dataset


def create_dataset_with_batch(image_dir, ann_dir, batch_size):
    image_list = os.listdir(image_dir)

    label_list = []
    bbox_list = []

    for filename in image_list:
        file_path = os.path.join(ann_dir, filename.replace('png', 'xml'))
        classes, bboxes = read_label(file_path, norm=True)

        length = len(bboxes)
        labels = classes

        if length < max_bbox_num:
            labels = labels + [999] * (max_bbox_num - length)
            for _ in range(max_bbox_num - length):
                bboxes.append([0., 0., 0., 0.])
        bbox_list.append(bboxes)
        label_list.append(labels)

    image_list = [os.path.join(image_dir, name) for name in image_list]
    dataset = dataset_gener_with_batch(image_list, bbox_list, label_list,batch_size)

    return dataset, len(image_list) // batch_size

def create_dataset_with_batch_json(image_dir, ann_json, batch_size):
    with open(ann_json, 'r') as f:
        gt_data = json.load(f)

    image_list = os.listdir(image_dir)
    label_list = []
    bbox_list = []
    for filename in image_list:
        classes = gt_data[filename.split('.')[0]]['class'].copy()
        bboxes = gt_data[filename.split('.')[0]]['box'].copy()
        for box in bboxes:
            box[0] = box[0] / cp_height
            box[1] = box[1] / cp_width
            box[2] = box[2] / cp_height
            box[3] = box[3] / cp_width

        length = len(bboxes)
        labels = classes

        if length < max_bbox_num:
            labels = labels + [999] * (max_bbox_num - length)
            for _ in range(max_bbox_num - length):
                bboxes.append([0., 0., 0., 0.])
        bbox_list.append(bboxes)
        label_list.append(labels)


    image_list = [os.path.join(image_dir, name) for name in image_list]
    dataset = dataset_gener_with_batch(image_list, bbox_list, label_list, batch_size)
    return dataset, len(image_list) // batch_size











































