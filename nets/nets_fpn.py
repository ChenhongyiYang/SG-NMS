import tensorflow as tf
from nets import resnet_v1
from nets import tf_ops as tfo

import tensorflow.contrib.slim as slim



def rpn_net(inputs, anchor_num, batch_size, scope='rpn'):
    '''
    :param inputs:
    :param anchor_num:
    :param batch_size:
    :param scope:
    :return: cls_pred: [batch_size, N, 2]
    :return: cls_logits: [batch_size, N, 2]
    :return: loc_pred: [batch_size, N, 4]
    '''
    with tf.variable_scope(scope, reuse=None):
        net = inputs
        net = slim.conv2d(net, 512, [3,3], scope='conv3x3')

        num_cls_pre = anchor_num * 2
        cls_pred = slim.conv2d(net, num_cls_pre, [1,1], activation_fn=None, scope='cls_pred')

        num_loc_pre = anchor_num * 4
        loc_pred = slim.conv2d(net, num_loc_pre, [1,1], activation_fn=None, scope='loc_pred')

        cls_logits = tf.reshape(cls_pred, [batch_size,-1,2])
        loc_pred = tf.reshape(loc_pred, [batch_size,-1,4])
        return cls_logits, loc_pred


def rfcn_occ_asynchronous(inputs,
                 anchor_tensor: list,
                 anchor_num: list,
                 batch_size: int,
                 num_classes: int,
                 proposal_num: list,
                 Ks: list,
                 select_nms_th: float,
                 net,
                 scope='rfcn_occ'):

    with tf.variable_scope(scope, 'rfcn_occ', [inputs], reuse=None):
        rpn_logit_list = []
        rpn_loc_list = []
        rois_selected_list = []
        score_map_list = []
        loc_map_list = []
        nms_map_list = []
        attention_map_loc_list = []
        attention_map_cls_list = []
        attention_map_nms_list = []

        if net == 'resnet_50_v1':
            nets = resnet_v1.FPN_resnet_50_v1(inputs, prefix=scope + '/', is_training=True)
        elif net == 'resnet_101_v1':
            nets = resnet_v1.FPN_resnet_101_v1(inputs, prefix=scope + '/', is_training=True)
        elif net == 'resnet_50_v2':
            nets = resnet_v1.FPN_resnet_50_v2(inputs, prefix=scope + '/', is_training=True)
        elif net == 'resnet_101_v2':
            nets = resnet_v1.FPN_resnet_101_v2(inputs, prefix=scope + '/', is_training=True)
        else:
            raise ValueError('Invalid backbone!')

        for i, net in enumerate(nets):
            # rpn
            rpn_logits, rpn_loc = rpn_net(net, anchor_num[i], batch_size, scope='rpn_%d' % i)
            rpn_logits_stop = tf.stop_gradient(rpn_logits)
            rpn_loc_stop = tf.stop_gradient(rpn_loc)

            rpn_cls_score = tf.nn.softmax(rpn_logits_stop)[:, :, 1]

            rois_selected, rois_score = tfo.roi_select(rpn_cls_score, rpn_loc_stop, anchor_tensor[i], batch_size,
                                                       proposal_num[i], select_nms_th, with_score=True)
            rois_selected = tf.stop_gradient(rois_selected)

            # object detection
            score_map = score_net(net, num_classes * Ks[i][0] * Ks[i][1], scope='class_score_%d' % i)
            loc_map = score_net(net, 4 * Ks[i][0] * Ks[i][1], scope='loc_score_%d' % i)
            nms_map = score_net(net, 4 * Ks[i][0] * Ks[i][1], scope='nms_score_%d' % i)

            attention_map_loc = score_net(net, 1 * Ks[i][0] * Ks[i][1], scope='attention_map_loc_%d' % i)
            attention_map_cls = score_net(net, 1 * Ks[i][0] * Ks[i][1], scope='attention_map_cls_%d' % i)
            attention_map_nms = score_net(net, 1 * Ks[i][0] * Ks[i][1], scope='attention_map_nms_%d' % i)


            rpn_logit_list.append(rpn_logits)
            rpn_loc_list.append(rpn_loc)
            rois_selected_list.append(rois_selected)
            score_map_list.append(score_map)
            loc_map_list.append(loc_map)
            nms_map_list.append(nms_map)
            attention_map_loc_list.append(attention_map_loc)
            attention_map_cls_list.append(attention_map_cls)
            attention_map_nms_list.append(attention_map_nms)


        return [rpn_logit_list, rpn_loc_list, rois_selected_list, score_map_list, loc_map_list, nms_map_list,
                attention_map_loc_list, attention_map_cls_list, attention_map_nms_list]


def score_net(inputs, num_channels, scope, reuse=None):
    with tf.variable_scope(scope,reuse=reuse):
        net = inputs
        net = slim.conv2d(net, 256, [3,3])
        net = slim.conv2d(net, 256, [3,3])
        net = slim.conv2d(net, 256, [3,3])
        net = slim.conv2d(net, num_channels, [1,1], activation_fn=None, scope='score_pred')
        return net


def roi_align_score(score_map, rois, K, num_class, batch_size, proposal_num, redu='mean', scope='position_sensitive_roi_align'):
    '''
    :param score_map:  [batch_size, feat_shape[0], feat_shape[1], num_channels], num_channels = num_class * K^2
    :param rois: [batch_size, proposal_num, 4]
    :param K: K: (K0, K1)
    :param num_class: int
    :param batch_size: int
    :param proposal_num: int
    :param redu: string
    :param scope: string
    :return:
    '''
    with tf.name_scope(scope):
        rois = tf.stop_gradient(rois)
        k0, k1 = K
        position_scores = []
        for i in range(k0):
            for j in range(k1):
                ymin = rois[:,:,0] + tf.div((rois[:,:,2] - rois[:,:,0]), k0) * i
                ymax = rois[:,:,0] + tf.div((rois[:,:,2] - rois[:,:,0]), k0) * (i + 1.)
                xmin = rois[:,:,1] + tf.div((rois[:,:,3] - rois[:,:,1]), k1) * j
                xmax = rois[:,:,1] + tf.div((rois[:,:,3] - rois[:,:,1]), k1) * (j + 1.)
                rois_k = tf.stack([ymin,xmin,ymax,xmax], axis=-1) #[batch_size, proposal_num, 4]
                map_k = score_map[:,:,:,(i*k1+j)*num_class:(i*k1+j+1)*num_class] #[batch_size, feat_shape[0], feat_shape[1], num_class]
                batch_scores = []
                for b in range(batch_size):
                    batch_rois = rois_k[b,:,:] #[proposal_num,4]
                    batch_map = map_k[b,:,:,:] #[feat_shape[0], feat_shape[1], num_classes]
                    batch_map = tf.expand_dims(batch_map, axis=0) #[1, feat_shape[0], feat_shape[1], num_classes]
                    box_inds = tf.constant([0] * proposal_num, dtype=tf.int32) #[proposal_num]
                    roi_aligned = tf.image.crop_and_resize(batch_map,batch_rois,box_inds,(1,1)) #[proposal_num,1,1,num_classes]
                    roi_aligned = tf.reshape(roi_aligned, [-1, num_class, 1]) #[proposal_num, num_class, 1]
                    roi_aligned = tf.expand_dims(roi_aligned, axis=0) #[1, proposal_num, num_class, 1]
                    batch_scores.append(roi_aligned)
                position_score = tf.concat(batch_scores, axis=0) #[batch_size, proposal_num, num_class, 1]
                position_scores.append(position_score)
        logits = tf.concat(position_scores,axis=3) #[batch_size, proposal_num, num_class, 9]
        if redu == 'mean':
            logits = tf.reduce_mean(logits, axis=-1)
        elif redu == 'max':
            logits = tf.reduce_max(logits, axis=-1)
        elif redu == 'none':
            logits = logits
        else:
            logits = tf.reduce_mean(logits, axis=-1)
        return logits