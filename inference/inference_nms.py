import tensorflow as tf
import numpy as np
import time
import os
import shutil
from tqdm import tqdm
import cv2

from nets import net_class
from nms import nms_ae

slim = tf.contrib.slim



class Tester(object):
    def __init__(self, net_param: net_class.nets_params, img_shape, process_fn, model, out_txt_dir):
        self.process_fn = process_fn
        self.net_param = net_param
        self.nets = net_class.Nets(self.net_param)
        self.model = model
        self.out_txt_dir = out_txt_dir
        self.img_shape = img_shape
        self.class_num = self.net_param.num_classes

    def set_dir(self, out_dir):
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.mkdir(out_dir)

    def read_img(self, img_place):
        img_string = tf.read_file(img_place)
        img = tf.image.decode_png(img_string)
        img = tf.cast(img, dtype=tf.float32)
        img = tf.div(img, 255.)
        _image = self.process_fn(img, self.net_param.net_shape)
        image_4d = tf.expand_dims(_image, axis=0)
        return image_4d

    def get_test_imgs(self, dir):
        file_list = os.listdir(dir)
        for i, f in enumerate(file_list):
            file_list[i] = os.path.join(dir, f)
        return file_list

    def post_process(self, roi_result_all_layer,
                     class_result_all_layer,
                     scores_result_all_layer,
                     ae_result_all_layer,
                     nms_mode,
                     ths, Nts,
                     second_nms, th2, Nt2,
                     sigma, ae_dis):

        if second_nms:
            nms_box_all_layer = []
            nms_score_all_layer = []
            nms_class_all_layer = []

            layer_num = len(roi_result_all_layer)
            for l in range(layer_num):
                nms_boxes, nms_scores, nms_classes = nms_ae.nms_all_class(np.array(roi_result_all_layer[l]),
                                                                          np.array(scores_result_all_layer[l]),
                                                                          np.array(class_result_all_layer[l]),
                                                                          np.array(ae_result_all_layer[l]),
                                                                          class_num=2,
                                                                          mode=nms_mode,
                                                                          threshold=ths[l], Nt=Nts[l],
                                                                          sigma=sigma, ae_dis=ae_dis[l])
                nms_box_all_layer = nms_box_all_layer + nms_boxes
                nms_score_all_layer = nms_score_all_layer + nms_scores
                nms_class_all_layer = nms_class_all_layer + nms_classes

            if not len(nms_box_all_layer) > 0:
                return np.array([]), np.array([]), np.array([])
            nms_boxes_2, nms_scores_2, nms_classes_2 = nms_ae.second_nms_all_class(nms_box_all_layer,
                                                                                   nms_score_all_layer,
                                                                                   nms_class_all_layer,
                                                                                   class_num=2,
                                                                                   Nt=Nt2, threshold=th2)
            if len(nms_boxes_2) > 0:
                final_boxes = nms_boxes_2[0]
                final_scores = nms_scores_2[0]
                final_classes = nms_classes_2[0]
                return final_boxes, final_scores, final_classes
            else:
                return np.array([]), np.array([]), np.array([])
        else:
            box_flatten = np.concatenate(roi_result_all_layer, axis=0)
            score_flatten = np.concatenate(scores_result_all_layer, axis=0)
            class_flatten = np.concatenate(class_result_all_layer, axis=0)
            ae_flatten = np.concatenate(ae_result_all_layer, axis=0)
            nms_boxes, nms_scores, nms_classes = nms_ae.nms_all_class(np.array(box_flatten),
                                                                      np.array(score_flatten),
                                                                      np.array(class_flatten),
                                                                      np.array(ae_flatten),
                                                                      class_num=2,
                                                                      mode=nms_mode,
                                                                      threshold=ths[0], Nt=Nts[0],
                                                                      sigma=sigma, ae_dis=ae_dis[0])

            if not len(nms_boxes) > 0:
                return np.array([]), np.array([]), np.array([])
            else:
                final_boxes = np.array(nms_boxes[0])
                final_scores = np.array(nms_scores[0])
                final_class = np.array(nms_classes[0])
                return final_boxes, final_scores, final_class




    def record(self, out_name, boxes, scores, classes, ih, iw):
        fw = open(out_name, 'w')
        n = boxes.shape[0]
        if n == 0:
            fw.close()
            return
        boxes = boxes * np.array([[ih, iw, ih, iw]])
        boxes = np.round(boxes)

        for j in range(n):
            fw.write('%d %d %d %d %d %.5f\n' % (
                classes[j],
                boxes[j, 0], boxes[j, 1], boxes[j, 2], boxes[j, 3],
                scores[j]))
        fw.close()

    def record_list(self, out_name, box_list, class_list, score_list, ae_list, ih, iw):
        fw = open(out_name, 'w')
        layer_num = len(box_list)

        for l in range(layer_num):
            box_l = box_list[l]
            box_l = box_l * np.array([[ih, iw, ih, iw]])

            score_l = score_list[l]
            class_l = class_list[l]
            ae_l = ae_list[l]

            n = box_l.shape[0]
            for j in range(n):
                line = '%d %d %d %d %d %.5f %d %.5f' % (
                class_l[j], box_l[j, 0], box_l[j, 1], box_l[j, 2], box_l[j, 3], score_l[j], l, ae_l[j])
                fw.write('%s\n' % line)
        fw.close()

    def run(self, test_img_dir, nms_param, is_list, eval_num=0):
        file_list = self.get_test_imgs(test_img_dir)
        if eval_num == 0:
            test_num = len(file_list)
        else:
            test_num = eval_num

        self.set_dir(self.out_txt_dir)

        tf.reset_default_graph()
        img_place = tf.placeholder(tf.string)
        image_4d = self.read_img(img_place)
        rois_all_layer, class_all_layer, scores_all_layer, ae_all_layer = self.nets.get_test_ops(image_4d)

        layer_num = len(rois_all_layer)

        tensor_to_run = rois_all_layer + class_all_layer + scores_all_layer + ae_all_layer

        saver = tf.train.Saver()

        start_time = time.time()
        with tf.Session() as sess:
            saver.restore(sess, self.model)
            for i in tqdm(range(test_num), desc='Inference'):
                img_name = file_list[i]
                img = cv2.imread(img_name)
                ih, iw = img.shape[0:2]
                results = sess.run(tensor_to_run, feed_dict={img_place: img_name})
                roi_result_all_layer = list(results[:layer_num])
                class_result_all_layer = list(results[layer_num:layer_num * 2])
                scores_result_all_layer = list(results[layer_num * 2:layer_num * 3])
                ae_result_all_layer = list(results[layer_num * 3:layer_num * 4])

                final_boxes, final_scores, final_classes = self.post_process(roi_result_all_layer,
                                                                             class_result_all_layer,
                                                                             scores_result_all_layer,
                                                                             ae_result_all_layer,
                                                                             nms_mode=nms_param['mode'],
                                                                             ths=nms_param['ths'], Nts=nms_param['Nts'],
                                                                             second_nms=nms_param['second_nms'], th2=nms_param['th2'], Nt2=nms_param['Nt2'],
                                                                             sigma=nms_param['sigma'], ae_dis=nms_param['ae_dis'])

                out_name = os.path.join(self.out_txt_dir, os.path.split(img_name)[-1].replace('png', 'txt'))
                if not is_list:
                    self.record(out_name, final_boxes, final_scores, final_classes, ih, iw)
                else:
                    self.record_list(out_name, roi_result_all_layer, class_result_all_layer, scores_result_all_layer, ae_result_all_layer, ih, iw)
        print('Inference complete! FPS:%.2f' % (test_num / (time.time() - start_time)))































