import tensorflow as tf
import os
from preprocessing import tf_image
from random import shuffle
from kitti import kitti_eval
import cv2

KITTI_VEHICLES = ['Car', 'Truck', 'Van', 'Tram','Pedestrian','Cyclist']
KITTI_VEHICLES2 = ['Car','Van','Pedestrian','Cyclist']
KITTI_CARS = ['Car', 'Van']


k_height, k_width = 375, 1242

max_bbox_num = 22

def class_encode(c):
    if c == 'Car' or c == 'Van':
        return 1
    elif c == 'Pedestrian':
        return 2
    elif c == 'Cyclist':
        return 3


def class_encode_car(c):
    if c in KITTI_CARS:
        return 1
    else:
        return 0


def read_kitti_label(label_file, encode=False, norm=False, only_car=True):
    f = open(label_file, 'r')
    lines = f.readlines()
    f.close()
    classes = []
    bboxes = []
    zs = []
    for line in lines:
        line = line.split(' ')
        if len(line) == 0:
            continue
        cclass = line[0]
        if cclass not in KITTI_VEHICLES2:
            continue
        if only_car and class_encode_car(cclass) != 1:
            continue
        if encode:
            classes.append(class_encode_car(cclass))
        else:
            classes.append(cclass)
        if norm:
            bbox = [float(line[5]) / k_height, float(line[4]) / k_width, float(line[7]) / k_height, float(line[6]) / k_width]
        else:
            bbox = [float(line[5]), float(line[4]), float(line[7]), float(line[6])]
        bboxes.append(bbox)
        zs.append(float(line[13]))
    classes, bboxes, zs = sort_by_z(zs, classes, bboxes)
    return classes, bboxes, zs




def read_kitti_label_2(label_file, h, w, encode=False, norm=False, only_car=True):
    f = open(label_file, 'r')
    lines = f.readlines()
    f.close()
    classes = []
    bboxes = []
    for line in lines:
        line = line.split(' ')
        if len(line) == 0:
            continue
        cclass = line[0]
        if cclass not in KITTI_VEHICLES2:
            continue
        if only_car and class_encode_car(cclass) != 1:
            continue
        if encode:
            classes.append(class_encode_car(cclass))
        else:
            classes.append(cclass)
        if norm:
            bbox = [float(line[5]) / h, float(line[4]) / w, float(line[7]) / h,float(line[6]) / w]
        else:
            bbox = [float(line[5]), float(line[4]), float(line[7]), float(line[6])]
        bboxes.append(bbox)
    return classes, bboxes

def sort_by_z(zs,classes,bboxes):
    n = len(zs)
    inds = list(range(n))
    for i in range(n-1,-1,-1):
        for j in range(i):
            if zs[j] < zs[j+1]:
                temp = zs[j]
                zs[j] = zs[j+1]
                zs[j+1] = temp

                temp = inds[j]
                inds[j] = inds[j+1]
                inds[j+1] = temp
    new_classes = []
    new_bboxes = []
    for i in range(n):
        new_bboxes.append(bboxes[inds[i]])
        new_classes.append(classes[inds[i]])
    return new_classes, new_bboxes, zs


def dataset_gener_with_batch(image_list, bbox_list, label_list,batch_size):
    def _parse_func(filename):
        img_string = tf.read_file(filename)
        img = tf.image.decode_png(img_string)
        img = tf_image.resize_image(img, (k_height,k_width),
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
    shuffle(image_list)

    label_list = []
    bbox_list = []

    f = open('/scratch/ChenhongyiYang/kitti_shape.txt')
    lines = f.readlines()
    f.close()

    shape_dict = {}
    for line in lines:
        info = line.strip().split(' ')
        fname, h, w = info[0], float(info[1]), float(info[2])
        shape_dict[fname] = (h, w)


    for filename in image_list:
        file_path = os.path.join(ann_dir, filename.replace('png', 'txt'))
        h, w = shape_dict[filename.split('.')[0]]
        classes, bboxes = read_kitti_label_2(file_path, h, w, True, True, True)

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
    return dataset, len(image_list)//batch_size



























