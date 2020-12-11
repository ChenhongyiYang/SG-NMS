import tensorflow as tf
from nets import nets_fpn
from nets import tf_ops
from nets import loss_function
from nets import custom_layers
from collections import namedtuple
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import math_ops

nets_params = namedtuple('net_parameters', ['net_shape',  # tuple
                                            'img_shape',
                                            'crop_shape',
                                            'num_classes',  # int
                                            'layer_num',  # int
                                            'feat_shape',  # list of tuples
                                            'anchor_sizes',  # list of tuples
                                            'anchor_ratios',  # list of tuples
                                            'anchor_offset',  # float
                                            'proposal_nums',  # list of int
                                            'Ks',  # list of tuple
                                            'max_bbox_num',  # int
                                            'anchor_nums',  # list of int
                                            'rpn_pos_threshold',  # float
                                            'rpn_neg_threshold', #float
                                            'rpn_neg_threshold_low', #float
                                            'net_loc_threshold', #float
                                            'net_pos_threshold',  # float
                                            'net_neg_threshold',  # float
                                            'net_neg_threshold_low', #float
                                            'ae_up_threshold',
                                            'ae_low_threshold',
                                            'select_nms_th',  # float
                                            'noise_xy',
                                            'noise_wh',
                                            'back_OHEM',
                                            'rpn_mini_batch',
                                            'rfcn_mini_batch',
                                            'output_box_ratio_high',
                                            'output_box_ratio_low',
                                            'out_box_num',
                                            'net'  # float
                                            ])


class Nets(object):
    def __init__(self, params: nets_params):
        self.params = params

    def anchor_gener_auto(self):
        anchors_list = []
        for i in range(self.params.layer_num):
            anchors = tf_ops.anchor_gener(self.params.net_shape,
                                          self.params.feat_shape[i],
                                          self.params.anchor_sizes[i],
                                          self.params.anchor_ratios[i],
                                          self.params.anchor_nums[i])
            anchors_list.append(anchors)
        return anchors_list

    def anchor_tensor_gener(self, anchors_list):
        anchor_tensor_list = []
        for i in range(len(self.params.feat_shape)):
            anchor_tensor = tf_ops.anchor_tensor_gener(anchors_list[i])
            anchor_tensor_list.append(anchor_tensor)
        return anchor_tensor_list

    def get_anchor(self):
        anchors_list = self.anchor_gener_auto()
        anchor_tensor_list = self.anchor_tensor_gener(anchors_list)
        return anchors_list, anchor_tensor_list


    def rfcn_occ_asynchronous(self, inputs, anchor_tensor_list, batch_size, is_training, new=False):
        arg_scope = self.rfcn_argscope(is_training)
        with slim.arg_scope(arg_scope):
            return nets_fpn.rfcn_occ_asynchronous(inputs,
                                            anchor_tensor_list,
                                            self.params.anchor_nums,
                                            batch_size,
                                            self.params.num_classes,
                                            self.params.proposal_nums,
                                            self.params.Ks,
                                            self.params.select_nms_th,
                                            self.params.net)

    def data_prepare(self, images, labels, bboxes, anchors_list, pre_fn, batch_size):
        image_4d = []

        labels_nets_4d = []
        bboxes_nets_4d = []

        glabels_rpn_4d_list = [[] for _ in range(self.params.layer_num)]
        glocs_rpn_4d_list = [[] for _ in range(self.params.layer_num)]
        gscores_rpn_4d_list = [[] for _ in range(self.params.layer_num)]

        for i in range(batch_size):
            image_, labels_, bboxes_ = pre_fn(images[i, :], labels[i, :], bboxes[i, :], self.params.img_shape,
                                              self.params.crop_shape, self.params.net_shape)
            pad_num = tf.subtract(self.params.max_bbox_num, tf.shape(labels_)[0])
            padding = tf.stack([0, pad_num])
            padding_label = tf.reshape(padding, [1, 2])
            padding_boxes = tf.stack([padding, [0, 0]], axis=0)
            labels_ = tf.pad(labels_, padding_label, constant_values=999)
            bboxes_ = tf.pad(bboxes_, padding_boxes)
            labels_nets_4d.append(tf.expand_dims(labels_, 0))
            bboxes_nets_4d.append(tf.expand_dims(bboxes_, 0))
            image_4d.append(tf.expand_dims(image_, axis=0))

            for j in range(self.params.layer_num):
                up_th, low_th = 99999., -1.
                glabels_, glocs_, gscores_ = tf_ops.bboxes_encode(anchors_list[j],
                                                                  bboxes_,
                                                                  labels_,
                                                                  up_th,
                                                                  low_th,
                                                                  self.params.net_shape,
                                                                  self.params.num_classes)

                glabels_rpn_4d_list[j].append(tf.expand_dims(glabels_, axis=0))
                glocs_rpn_4d_list[j].append(tf.expand_dims(glocs_, axis=0))
                gscores_rpn_4d_list[j].append(tf.expand_dims(gscores_, axis=0))

        if batch_size > 1:
            for i in range(self.params.layer_num):
                glabels_rpn_4d_list[i] = tf.concat(glabels_rpn_4d_list[i], axis=0)
                glocs_rpn_4d_list[i] = tf.concat(glocs_rpn_4d_list[i], axis=0)
                gscores_rpn_4d_list[i] = tf.concat(gscores_rpn_4d_list[i], axis=0)

            image_4d = tf.concat(image_4d, axis=0)

            labels_nets_4d = tf.concat(labels_nets_4d, axis=0)
            bboxes_nets_4d = tf.concat(bboxes_nets_4d, axis=0)
        else:
            for i in range(self.params.layer_num):
                glabels_rpn_4d_list[i] = glabels_rpn_4d_list[i]
                glocs_rpn_4d_list[i] = glocs_rpn_4d_list[i]
                gscores_rpn_4d_list[i] = gscores_rpn_4d_list[i]

            image_4d = image_4d[0]
            labels_nets_4d = labels_nets_4d[0]
            bboxes_nets_4d = bboxes_nets_4d[0]

        return [image_4d, glabels_rpn_4d_list, glocs_rpn_4d_list, gscores_rpn_4d_list, labels_nets_4d, bboxes_nets_4d]

    def compute_output_with_att(self, score_map, att_map, boxes, K, output_c, box_num, batch_size):
        feature_crop = nets_fpn.roi_align_score(score_map, boxes, K, output_c, batch_size, box_num, redu='none')
        att_crop = nets_fpn.roi_align_score(att_map, boxes, K, 1, batch_size, box_num, redu='none')
        spatial_att = slim.softmax(att_crop)
        output = tf.reduce_sum(spatial_att * feature_crop, axis=-1)
        return output

    def get_train_ops(self, images, labels, bboxes, pre_fn, batch_size):
        ############################################################################################################################################################################################
        # data prepare
        ############################################################################################################################################################################################
        labels = tf.cast(labels, tf.int64)
        bboxes = tf.cast(bboxes, tf.float32)
        anchors_list, anchor_tensor_list = self.get_anchor()
        prepare_outputs = self.data_prepare(images, labels, bboxes, anchors_list, pre_fn, batch_size)

        image_4d = prepare_outputs[0]
        glabels_rpn_4d_list = prepare_outputs[1]
        glocs_rpn_4d_list = prepare_outputs[2]
        gscores_rpn_4d_list = prepare_outputs[3]
        labels_net_4d = prepare_outputs[4]
        bboxes_net_4d = prepare_outputs[5]

        ############################################################################################################################################################################################
        # build model
        ############################################################################################################################################################################################
        model_outputs = self.rfcn_occ_asynchronous(image_4d, anchor_tensor_list, batch_size, is_training=True)

        rpn_logit_list = model_outputs[0]
        rpn_loc_list = model_outputs[1]
        rois_selected_list = model_outputs[2]
        score_map_list = model_outputs[3]
        loc_map_list = model_outputs[4]
        nms_map_list = model_outputs[5]
        attention_map_loc_list = model_outputs[6]
        attention_map_cls_list = model_outputs[7]
        attention_map_nms_list = model_outputs[8]

        ############################################################################################################################################################################################
        # RPN loss
        ############################################################################################################################################################################################
        rpn_loss = tf.constant(0., tf.float32)
        recall_rpn = tf.constant(0., tf.float32)
        precision_rpn = tf.constant(0., tf.float32)
        recall_div_rpn = tf.constant(0., tf.float32)
        precision_div_rpn = tf.constant(0., tf.float32)

        for i in range(self.params.layer_num):
            rpn_loss_i, recall_rpn_i, precision_rpn_i, rpn_loc_loss_i = loss_function.rpn_loss(rpn_logit_list[i],
                                                                                               rpn_loc_list[i],
                                                                                               glabels_rpn_4d_list[i],
                                                                                               glocs_rpn_4d_list[i],
                                                                                               gscores_rpn_4d_list[i],
                                                                                               batch_size,
                                                                                               pos_threshold=self.params.net_pos_threshold,
                                                                                               neg_threshold=self.params.net_neg_threshold,
                                                                                               neg_threshold_low=self.params.net_neg_threshold_low,
                                                                                               mini_batch_size=self.params.rpn_mini_batch)
            rpn_loss += rpn_loss_i
            recall_rpn += tf.cast(tf.not_equal(recall_rpn_i, -1.), tf.float32) * recall_rpn_i
            precision_rpn += tf.cast(tf.not_equal(precision_rpn_i, -1.), tf.float32) * precision_rpn_i
            recall_div_rpn += tf.cast(tf.not_equal(recall_rpn_i, -1.), tf.float32) * 1.
            precision_div_rpn += tf.cast(tf.not_equal(precision_rpn_i, -1.), tf.float32) * 1.


        ret_recall_rpn = tf.cond(recall_div_rpn > 0,
                             lambda: recall_rpn / recall_div_rpn,
                             lambda: tf.constant(0.0))
        ret_precision_rpn = tf.cond(precision_div_rpn > 0,
                                lambda: precision_rpn / precision_div_rpn,
                                lambda: tf.constant(0.0))


        ############################################################################################################################################################################################
        # localization loss
        ############################################################################################################################################################################################

        loc_loss = tf.constant(0., tf.float32)
        loc_pred_list = []
        for i in range(self.params.layer_num):
            up_th, low_th = 99999., -1.
            gt_labels_loc, gt_delta_loc, gt_scores_loc, gt_location_loc, _, _, _, _ = tf_ops.roi_encode(rois_selected_list[i],
                                                                                                          bboxes_net_4d,
                                                                                                          labels_net_4d,
                                                                                                          up_th,
                                                                                                          low_th,
                                                                                                          self.params.net_shape,
                                                                                                          self.params.num_classes,
                                                                                                          self.params.max_bbox_num)

            loc_pred = self.compute_output_with_att(loc_map_list[i], attention_map_loc_list[i], rois_selected_list[i],
                                                    self.params.Ks[i], 4, self.params.proposal_nums[i], batch_size)
            loc_pred_list.append(loc_pred)

            loc_loss_i = loss_function.single_loc_loss(loc_pred,
                                                       gt_labels_loc, gt_delta_loc, gt_scores_loc,
                                                       match_threshold=self.params.net_loc_threshold)
            loc_loss += loc_loss_i

        ############################################################################################################################################################################################
        # classification loss
        ############################################################################################################################################################################################
        cls_loss = tf.constant(0., tf.float32)
        recall = tf.constant(0., tf.float32)
        precision = tf.constant(0., tf.float32)
        recall_div = tf.constant(0., tf.float32)
        precision_div = tf.constant(0., tf.float32)
        good_num = tf.constant(0.0, tf.float32)
        pull_loss = tf.constant(0.0, tf.float32)
        push_loss = tf.constant(0.0, tf.float32)

        for i in range(self.params.layer_num):
            # ------------------------------------------
            #           bounding box decode
            # ------------------------------------------
            up_th, low_th = 99999., -1.
            bboxes_decoded = tf_ops.bboxes_decode(rois_selected_list[i], loc_pred_list[i])

            # apply random distortion
            bboxes_decoded = tf_ops.apply_random_distortion(bboxes_decoded, batch_size, self.params.proposal_nums[i], r_xy=self.params.noise_xy, r_wh=self.params.noise_wh)

            bboxes_decoded = tf.stop_gradient(bboxes_decoded)

            # ------------------------------------------
            #           classification encode
            # ------------------------------------------
            gt_labels_cls, gt_delta_cls, gt_scores_cls, gt_location_cls, \
            gt_labels_cls2, gt_delta_cls2, gt_scores_cls2, gt_location_cls2 = tf_ops.roi_encode(bboxes_decoded,
                                                                                                  bboxes_net_4d,
                                                                                                  labels_net_4d,
                                                                                                  up_th,
                                                                                                  low_th,
                                                                                                  self.params.net_shape,
                                                                                                  self.params.num_classes,
                                                                                                  self.params.max_bbox_num)

            cls_logits = self.compute_output_with_att(score_map_list[i], attention_map_cls_list[i], bboxes_decoded,
                                                      self.params.Ks[i], self.params.num_classes, self.params.proposal_nums[i], batch_size)

            # ---------------------------------------
            #          classification loss
            # ---------------------------------------

            cls_loss_i, recall_i, precision_i = loss_function.single_cls_loss(cls_logits,
                                                                              gt_labels_cls, gt_scores_cls,
                                                                              batch_size,
                                                                              class_num=self.params.num_classes,
                                                                              pos_threshold=self.params.net_pos_threshold,
                                                                              neg_threshold=self.params.net_neg_threshold,
                                                                              neg_threshold_low=self.params.net_neg_threshold_low,
                                                                              mini_batch_size=self.params.rfcn_mini_batch,
                                                                              OHEM=True,
                                                                              back_OHEM=nets_params.back_OHEM
                                                                              )
            cls_loss += cls_loss_i
            recall += tf.cast(tf.not_equal(recall_i, -1.), tf.float32) * recall_i
            precision += tf.cast(tf.not_equal(precision_i, -1.), tf.float32) * precision_i
            recall_div += tf.cast(tf.not_equal(recall_i, -1.), tf.float32) * 1.
            precision_div += tf.cast(tf.not_equal(precision_i, -1.), tf.float32) * 1.

            good_mask_i = tf.greater_equal(gt_scores_cls, self.params.net_pos_threshold)
            good_num_i = tf.reduce_mean(tf.reduce_sum(tf.cast(good_mask_i, tf.float32), axis=1), axis=0)
            good_num = good_num + good_num_i

            # ---------------------------------------
            #               NMS Loss
            # ---------------------------------------
            nms_logits_feat = self.compute_output_with_att(nms_map_list[i], attention_map_nms_list[i], bboxes_decoded,
                                                           self.params.Ks[i], 4, self.params.proposal_nums[i], batch_size)

            nms_logits_gt = self.compute_output_with_att(nms_map_list[i], attention_map_nms_list[i], gt_location_cls,
                                                           self.params.Ks[i], 4, self.params.proposal_nums[i], batch_size)

            nms_logits_gt2 = self.compute_output_with_att(nms_map_list[i], attention_map_nms_list[i], gt_location_cls2,
                                                         self.params.Ks[i], 4, self.params.proposal_nums[i], batch_size)

            pull_loss_i, push_loss_i = loss_function.single_nms_loss(bboxes_decoded,
                                                                     gt_scores_cls, gt_location_cls,
                                                                     gt_scores_cls2, gt_location_cls2,
                                                                     nms_logits_feat, nms_logits_gt, nms_logits_gt2,
                                                                     th1=self.params.ae_up_threshold, th2=self.params.ae_low_threshold)
            pull_loss += pull_loss_i
            push_loss += push_loss_i

        # -----------------------------------------------------------------
        #               post process for some supervision
        # -----------------------------------------------------------------
        ret_recall = tf.cond(recall_div > 0,
                             lambda : recall / recall_div,
                             lambda : tf.constant(0.0))
        ret_precision = tf.cond(precision_div > 0,
                             lambda : precision / precision_div,
                             lambda : tf.constant(0.0))
        good_num = good_num / self.params.layer_num

        ############################################################################################################################################################################################
        # for train
        ############################################################################################################################################################################################
        total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
        ret_dict = {
            'Total loss': total_loss,
            'RPN loss': rpn_loss,
            'RPN Recall': ret_recall_rpn,
            'RPN Precision': ret_precision_rpn,
            'Positive RoIs': good_num,
            'Classify loss': cls_loss,
            'Localization loss': loc_loss,
            'Recall': ret_recall,
            'Precision': ret_precision,
            'Pull loss': pull_loss,
            'Push loss': push_loss            
        }

        log_item = ['Total loss', 'RPN loss', 'RPN Recall', 'RPN Precision',
                    'Positive RoIs', 'Classify loss', 'Localization loss',
                    'Recall', 'Precision', 'Pull loss','Push loss']
        return ret_dict, log_item

    def rid_weird_roi(self, rois, low_th, up_th):
        if len(rois.get_shape()) == 3:
            h = rois[:, :, 2] - rois[:, :, 0]
            w = rois[:, :, 3] - rois[:, :, 1]

            h = h * self.params.net_shape[0]
            w = w * self.params.net_shape[1]

            ratio = tf.div(h, w)

            valid_mask = tf.logical_and(tf.greater(ratio, low_th), tf.less(ratio, up_th))
            return valid_mask
        else:
            h = rois[:, 2] - rois[:, 0]
            w = rois[:, 3] - rois[:, 1]

            h = h * self.params.net_shape[0]
            w = w * self.params.net_shape[1]

            ratio = tf.div(h, w)

            valid_mask = tf.logical_and(tf.greater(ratio, low_th), tf.less(ratio, up_th))
            return valid_mask

    def get_ae_id_inference(self, box, trans):
        ymin = box[:, 0]
        xmin = box[:, 1]
        ymax = box[:, 2]
        xmax = box[:, 3]

        y = (ymin + ymax) / 2.
        x = (xmin + xmax) / 2.
        h = ymax - ymin
        w = xmax - xmin

        yxhw = tf.stack((y,x,h,w), axis=-1)
        # trans = tf.ones_like(yxhw)
        ae = tf.reduce_sum(trans * yxhw, axis=-1)
        return ae

    def get_test_ops(self, image_4d):
        batch_size = 1
        anchors_list, anchor_tensor_list = self.get_anchor()
        ############################################################################################################################################################################################
        # build model
        ############################################################################################################################################################################################
        model_outputs = self.rfcn_occ_asynchronous(image_4d, anchor_tensor_list, batch_size, is_training=True)

        rpn_logit_list = model_outputs[0]
        rpn_loc_list = model_outputs[1]
        rois_selected_list = model_outputs[2]
        score_map_list = model_outputs[3]
        loc_map_list = model_outputs[4]
        nms_map_list = model_outputs[5]
        attention_map_loc_list = model_outputs[6]
        attention_map_cls_list = model_outputs[7]
        attention_map_nms_list = model_outputs[8]


        ############################################################################################################################################################################################
        # post process
        ############################################################################################################################################################################################

        rois_all_layer = []
        class_all_layer = []
        scores_all_layer = []
        ae_all_layer = []

        for i in range(self.params.layer_num):
            # -----------------------------------------------------------------
            #                       run localization
            # -----------------------------------------------------------------
            loc_pred = self.compute_output_with_att(loc_map_list[i], attention_map_loc_list[i], rois_selected_list[i],
                                                    self.params.Ks[i], 4, self.params.proposal_nums[i], batch_size)

            # -----------------------------------------------------------------
            #                      bounding box decode
            # -----------------------------------------------------------------
            objs_i = tf_ops.bboxes_decode(rois_selected_list[i], loc_pred)

            # -----------------------------------------------------------------
            #                     run classification
            # -----------------------------------------------------------------
            cls_logits = self.compute_output_with_att(score_map_list[i], attention_map_cls_list[i], objs_i,
                                                      self.params.Ks[i], self.params.num_classes, self.params.proposal_nums[i], batch_size)

            # -----------------------------------------------------------------
            #                     run nms transformation
            # -----------------------------------------------------------------
            nms_trans = self.compute_output_with_att(nms_map_list[i], attention_map_nms_list[i], objs_i,
                                                     self.params.Ks[i], 4, self.params.proposal_nums[i], batch_size)

            # -----------------------------------------------------------------
            #              get confidence and remove odd boxes
            # -----------------------------------------------------------------
            cls_pred_i = slim.softmax(cls_logits)
            cls_scores_i = cls_pred_i[:, :, 1]
            cls_scores_i = tf.reshape(cls_scores_i, [self.params.proposal_nums[i]])

            cls_pred_i = tf.ones_like(cls_scores_i)

            objs_i = tf.reshape(objs_i, [self.params.proposal_nums[i], 4])
            nms_trans_i = tf.reshape(nms_trans, [self.params.proposal_nums[i], 4])
            ae_i = self.get_ae_id_inference(objs_i, nms_trans_i)
            valid_mask_i = self.rid_weird_roi(objs_i, low_th=self.params.output_box_ratio_low, up_th=self.params.output_box_ratio_high)

            cls_scores_i = cls_scores_i * tf.cast(valid_mask_i, tf.float32)

            # -----------------------------------------------------------------
            #                     get top k boxes
            # -----------------------------------------------------------------

            _, gather_inds = tf.nn.top_k(cls_scores_i, k=self.params.out_box_num)
            objs_i = tf.gather(objs_i, gather_inds)
            cls_pred_i = tf.gather(cls_pred_i, gather_inds)
            cls_scores_i = tf.gather(cls_scores_i, gather_inds)
            ae_i = tf.gather(ae_i, gather_inds)

            objs_i = tf.reshape(objs_i, [-1, 4])
            cls_pred_i = tf.reshape(cls_pred_i, [-1])
            cls_scores_i = tf.reshape(cls_scores_i, [-1])
            #if i == 2:
            #    cls_scores_i = tf.minimum(cls_scores_i * 1.1, 1.0)
            ae_i = tf.reshape(ae_i, [-1])

            rois_all_layer.append(objs_i)
            class_all_layer.append(cls_pred_i)
            scores_all_layer.append(cls_scores_i)
            ae_all_layer.append(ae_i)

        return rois_all_layer, class_all_layer, scores_all_layer, ae_all_layer




