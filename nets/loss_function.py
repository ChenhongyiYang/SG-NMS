import tensorflow as tf
from nets import custom_layers, nets_fpn
import tensorflow.contrib.slim as slim
from nets import tf_ops
import math


def rpn_loss(logits, locaizations,
             gclasses, glocs, gscores,
             batch_size,
             pos_threshold,
             neg_threshold,
             neg_threshold_low,
             mini_batch_size,
             neg_ratio = 2.):
    '''
    :param logits: [batch_size, anchor_num, 2]
    :param locaizations:  [batch_size, anchor_num, 4]
    :param gclasses: [batch_size, feat_shape[0], feat_shape[1], anchor_num]
    :param glocs: [batch_size, feat_shape[0], feat_shape[1], anchor_num, 4]
    :param gscores: [batch_size, feat_shape[0], feat_shape[1], anchor_num]
    :param batch_size: int
    :param pos_threshold: float
    :param neg_threshold: float
    :param neg_threshold_low: float
    :param mini_batch_size: int
    :param neg_ratio: float
    :return:
    '''
    with tf.name_scope('rpn_loss'):

        logits = tf.reshape(logits, [batch_size, -1, 2])
        locaizations = tf.reshape(locaizations, [batch_size, -1, 4])
        gclasses = tf.reshape(gclasses, [batch_size, -1])
        glocs = tf.reshape(glocs, [batch_size, -1, 4])
        gscores = tf.reshape(gscores, [batch_size, -1])

        # compute accraucy

        pmask = tf.logical_and(gscores > pos_threshold, tf.not_equal(gclasses, -1))  # [batch_size, N]
        fpmask = tf.cast(pmask, tf.float32)

        gclasses_ = tf.cast(pmask, tf.int64)

        nmask = tf.logical_and(gscores < neg_threshold, tf.not_equal(gclasses, -1))
        nmask = tf.logical_and(nmask, gscores >= neg_threshold_low)

        pos_num = tf.reduce_sum(fpmask, axis=1)  # [batch_size]

        '''
        #accuracy computing 
        prob = slim.softmax(logits)  # [batch_size, proposal_num, 2]
        pre = tf.argmax(prob, axis=2)  # [batch_size, proposal_num]
        acc_pos = accurate_compute(pre, gclasses_, pmask, batch_size)
        acc_neg = accurate_compute(pre, gclasses_, nmask, batch_size)
        '''

        #recall and precision computing
        prob = slim.softmax(logits)
        pscore = prob[:,:,1]
        recall, precision = recall_precision_compute(pscore, gclasses_, 0.5, tf.logical_or(nmask, pmask))

        pos_train_num = mini_batch_size // (neg_ratio + 1.)  # scalar
        pos_train_num = tf.minimum(pos_num, pos_train_num)  # [batch_size]

        neg_train_num = mini_batch_size - pos_train_num  # [batch_size]

        train_batch_logits, train_batch_label = sample_cls_batch(logits, gclasses_, pmask, nmask, pos_train_num,
                                                             neg_train_num, batch_size)

        with tf.name_scope('cls_loss'):
            cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_batch_logits,
                                                                      labels=train_batch_label)
            cls_loss_mean = tf.reduce_mean(cls_loss)

        # Add localization loss: smooth L1, L2, ...
        with tf.name_scope('localization'):
            # Weights Tensor: positive mask + random negative.
            localizations = tf.reshape(locaizations, [-1, 4])
            glocs = tf.reshape(glocs, [-1, 4])
            lmask = tf.reshape(pmask, [-1])
            pinds = tf.where(lmask)
            pre_locs = tf.reshape(tf.gather(localizations, pinds), [-1, 4])
            gt_locs = tf.reshape(tf.gather(glocs, pinds), [-1, 4])
            loc_loss = tf.cond(tf.shape(gt_locs)[0] > 0,
                               lambda: tf.reduce_mean(
                                   tf.reduce_sum(custom_layers.abs_smooth(pre_locs - gt_locs), axis=1)),
                               lambda: tf.constant(0.0))

        total_loss = cls_loss_mean + loc_loss
        slim.losses.add_loss(total_loss)

        #return total_loss, acc_pos, acc_neg, loc_loss
        return total_loss, recall, precision, loc_loss


def obj_detection_loss(logits, localizations,
                         gclasses, glocs, gscores,
                         batch_size,
                         class_num,
                         pos_threshold,
                         neg_threshold,
                         neg_threshold_low,
                         alpha,
                         neg_ratio=3.,
                         mini_batch_size=64
                         ):
    '''
    :param logits: [batch_size, N, num_classes]
    :param localizations: [batch_size, N, 4]
    :param gclasses: [batch_size, N]
    :param glocs: [batch_size, N, 4]
    :param gscores: [batch_size, N]
    :param batch_size: int
    :param class_num: int
    :param pos_threshold: float
    :param neg_threshold: flpat
    :param neg_threshold_low: float
    :param alpha: float
    :param neg_ratio: float
    :param mini_batch_size: int
    :return:
    '''
    with tf.name_scope('obj_detection_loss'):
        gclasses = tf.stop_gradient(gclasses)
        glocs = tf.stop_gradient(glocs)
        gscores = tf.stop_gradient(gscores)

        pmask = tf.logical_and(gscores > pos_threshold, tf.not_equal(gclasses, -1))  # [batch_size, proposal_num]

        fpmask = tf.cast(pmask, tf.float32)  # [batch_size, proposal_num]
        ipmask = tf.cast(pmask, tf.int64)

        gclasses_ = gclasses * ipmask

        nmask = tf.logical_and(gscores < neg_threshold, tf.not_equal(gclasses, -1))
        nmask = tf.logical_and(nmask, gscores > neg_threshold_low)
        fnmask = tf.cast(nmask, tf.float32)  # [batch_size, proposal_num]

        pos_num = tf.reduce_sum(fpmask, axis=1)  # [batch_size]
        neg_num = tf.reduce_sum(fnmask, axis=1)  # [batch_size]

        prob = slim.softmax(logits)  # [batch_size, proposal_num, classes_num]
        pre = tf.argmax(prob, axis=2)  # [batch_size, proposal_num]
        acc_pos = accurate_compute(pre, gclasses_, pmask, batch_size)
        acc_neg = accurate_compute(pre, gclasses_, nmask, batch_size)

        pos_train_num = mini_batch_size // (neg_ratio + 1.)  # scalar
        pos_train_num = tf.minimum(pos_num, pos_train_num)  # [batch_size]

        neg_train_num = mini_batch_size - pos_train_num  # [batch_size]

        train_batch_logits, train_batch_label = sample_cls_batch(logits, gclasses_, pmask, nmask, pos_train_num,
                                                                 neg_train_num, batch_size)
        with tf.name_scope('cls_loss'):
            cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_batch_logits,
                                                                      labels=train_batch_label)
            cls_loss_mean = tf.reduce_mean(cls_loss)

            slim.losses.add_loss(cls_loss_mean)

        with tf.name_scope('localization_loss'):
            localizations = tf.reshape(localizations, [-1, 4])
            glocs = tf.reshape(glocs, [-1, 4])
            lmask = tf.reshape(pmask, [-1])
            pinds = tf.where(lmask)
            pre_locs = tf.reshape(tf.gather(localizations, pinds), [-1, 4])
            gt_locs = tf.reshape(tf.gather(glocs, pinds), [-1, 4])
            loc_loss = tf.cond(tf.shape(gt_locs)[0] > 0,
                               lambda: tf.reduce_mean(
                                   tf.reduce_sum(custom_layers.abs_smooth(pre_locs - gt_locs), axis=1)),
                               lambda: tf.constant(0.0))
            loc_loss_mean = loc_loss * alpha
            slim.losses.add_loss(loc_loss_mean)
    return cls_loss_mean, loc_loss,  acc_pos, acc_neg


def single_cls_loss(logits,
                    gclasses, gscores,
                    batch_size, class_num,
                    pos_threshold, neg_threshold, neg_threshold_low=0.,
                    neg_ratio=3., mini_batch_size=64,
                    OHEM=False, back_OHEM=False
                    ):
    '''
    :param logits: [batch_size, N, class_num]
    :param gclasses: [batch_size, N]
    :param gscores: [batch_size, N]
    :param batch_size: int
    :param class_num: int
    :param pos_threshold: float
    :param neg_threshold: float
    :param neg_threshold_low: float
    :param neg_ratio: float
    :param mini_batch_size: int
    :param OHEM: bool
    :return:
    '''
    with tf.name_scope('cls_loss'):


        gclasses = tf.stop_gradient(gclasses)
        gscores = tf.stop_gradient(gscores)

        on_value = 0.999
        off_value = (1. - on_value) / (class_num - 1)

        pmask = tf.logical_and(gscores > pos_threshold, tf.not_equal(gclasses, -1))  # [batch_size, proposal_num]

        fpmask = tf.cast(pmask, tf.float32)  # [batch_size, proposal_num]
        ipmask = tf.cast(pmask, tf.int64)

        gclasses_ = gclasses * ipmask

        nmask = tf.logical_and(gscores < neg_threshold, tf.not_equal(gclasses, -1))
        nmask = tf.logical_and(nmask, gscores >= neg_threshold_low)

        pos_num = tf.reduce_sum(fpmask, axis=1)  # [batch_size]

        ##########################################################################################
        # compute recall and precision, only for binary class_num == 2 !!!!!!
        ##########################################################################################
        prob = slim.softmax(logits)  # [batch_size, proposal_num, classes_num]
        pscore = prob[:,:,1]  # [batch_size, proposal_num]
        recall, precision = recall_precision_compute(pscore, gclasses_, 0.5, tf.logical_or(nmask, pmask))


        ##########################################################################################
        # extra negative loss
        ##########################################################################################
        bnum = 128
        bmask = tf.less(gscores, 0.1)
        #random sample
        if not back_OHEM:
            blogits, blabels = sample_instance(logits, gclasses_, bmask, tf.constant([bnum]*batch_size, tf.int32), batch_size, True)
        else:
            false_mask = tf.greater(gscores, 999.)
            blogits, blabels = OHEM_select(logits,gclasses_, bmask, false_mask, batch_size, class_num, bnum)

        neg_loss_mean = tf.cond(tf.shape(blogits)[0] > 0,
                                lambda: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                    logits=blogits,
                                    labels=tf.reshape(tf.one_hot(blabels, class_num, on_value=on_value, off_value=off_value), [-1, class_num]))),
                                lambda: tf.constant(0.0))
        slim.losses.add_loss(neg_loss_mean)

        ##########################################################################################
        # hard neg loss
        ##########################################################################################
        '''
        K = 128
        hard_neg_logits = hard_negative_select(gscores, logits, pos_threshold, class_num, batch_size, K)
        hard_neg_loss_mean = tf.cond(tf.shape(hard_neg_logits)[0] > 0,
                                lambda: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                    logits=hard_neg_logits,
                                    labels=tf.one_hot(tf.zeros(tf.shape(hard_neg_logits)[0], tf.int32), class_num, on_value=on_value, off_value=off_value))),
                                lambda: tf.constant(0.0))
        slim.losses.add_loss(hard_neg_loss_mean)
        '''


        ##########################################################################################
        # random sample
        ##########################################################################################
        if not OHEM:
            pos_train_num = mini_batch_size // (neg_ratio + 1.)  # scalar
            pos_train_num = tf.minimum(pos_num, pos_train_num)  # [batch_size]

            neg_train_num = mini_batch_size - pos_train_num  # [batch_size]

            train_batch_logits, train_batch_label = sample_cls_batch(logits, gclasses_, pmask, nmask, pos_train_num,
                                                                     neg_train_num, batch_size)
            with tf.name_scope('cls_loss'):
                cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_batch_logits,
                                                                          labels=train_batch_label)
                cls_loss_mean = tf.reduce_mean(cls_loss)

                slim.losses.add_loss(cls_loss_mean)

        ##########################################################################################
        # OHEM
        ##########################################################################################
        else:
            train_logits, train_label = OHEM_select(logits, gclasses_, pmask, nmask, batch_size, class_num, mini_batch_size)
            with tf.name_scope('cls_loss'):
                cls_loss_mean = tf.cond(tf.shape(train_logits)[0]>0,
                                        lambda : tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                            logits=train_logits,
                                            labels=tf.reshape(tf.one_hot(train_label, class_num, on_value=on_value, off_value=off_value), [-1, class_num]))),
                                        lambda : tf.constant(0.0))
                slim.losses.add_loss(cls_loss_mean)
        #return cls_loss_mean, acc_pos, acc_neg
        return cls_loss_mean, recall, precision


def single_loc_loss(delta,
                    gclasses, gdelta, gscores,
                    match_threshold):
    '''
    :param delta: [batch_size, N ,4]
    :param gclasses: [batch_size, N]
    :param gdelta: [batch_size, N, 4]
    :param gscores: [batch_size, N]
    :param match_threshold: float
    :return:
    '''

    with tf.name_scope('obj_detection_loss'):
        gclasses = tf.stop_gradient(gclasses)
        glocs = tf.stop_gradient(gdelta)
        gscores = tf.stop_gradient(gscores)

        pmask = tf.logical_and(gscores > match_threshold, tf.not_equal(gclasses, -1))  # [batch_size, proposal_num]

        with tf.name_scope('localization_loss'):
            delta = tf.reshape(delta, [-1, 4])
            glocs = tf.reshape(glocs, [-1, 4])
            lmask = tf.reshape(pmask, [-1])
            pinds = tf.where(lmask)
            pre_locs = tf.reshape(tf.gather(delta, pinds), [-1, 4])
            gt_locs = tf.reshape(tf.gather(glocs, pinds), [-1, 4])
            loc_loss_mean = tf.cond(tf.shape(gt_locs)[0] > 0,
                               lambda: tf.reduce_mean(
                                   tf.reduce_sum(custom_layers.abs_smooth(pre_locs - gt_locs), axis=1)),
                               lambda: tf.constant(0.0))
            loc_loss_mean = loc_loss_mean * 2.
            slim.losses.add_loss(loc_loss_mean)

    return loc_loss_mean


def single_nms_loss(rois,
                    gscores, glocation,
                    gscores2, glocation2,
                    trans_roi, trans_g1, trans_g2,
                    th1, th2):
    '''
    :param rois: [batch_size, N, 4]
    :param gscores: [batch_size, N]
    :param glocation: [batch_size, N, 4]
    :param gscores2: [batch_size, N]
    :param glocation2: [batch_size, N, 4]
    :param trans_roi: [batch_size, N, 4]
    :param trans_g1: [batch_size, N, 4]
    :param trans_g2: [batch_size, N, 4]
    :param th1: float
    :param th2: float
    :return:
    '''
    def get_ae_id(boxes, trans):
        y = (boxes[:, :, 0] + boxes[:, :, 2]) / 2.
        x = (boxes[:, :, 1] + boxes[:, :, 3]) / 2.
        h = boxes[:, :, 2] - boxes[:, :, 0]
        w = boxes[:, :, 3] - boxes[:, :, 1]
        feat_geo = tf.stack((y, x, h, w), axis=-1)
        feat_geo = tf.where(tf.is_nan(feat_geo), tf.zeros_like(feat_geo), feat_geo)
        feat_geo = tf.stop_gradient(feat_geo)  # [batch_size, N, 4]
        ae_id = tf.reshape(tf.reduce_sum(feat_geo * trans, axis=-1), [-1])
        return ae_id


    with tf.name_scope('nms_loss'):
        new_feat_geo = get_ae_id(rois, trans_roi)
        new_g_geo = get_ae_id(glocation, trans_g1)
        new_g_geo2 = get_ae_id(glocation2, trans_g2)

        pull_mask = tf.reshape(tf.greater(gscores, th1), [-1])
        push_mask = tf.reshape(tf.logical_and(tf.greater(gscores, th1), tf.greater(gscores2, th2)), [-1])

        pull_ind = tf.where(pull_mask)
        push_ind = tf.where(push_mask)

        pull_feat = tf.reshape(tf.gather(new_feat_geo, pull_ind), [-1])
        pull_gt = tf.reshape(tf.gather(new_g_geo, pull_ind), [-1])

        push_feat = tf.reshape(tf.gather(new_feat_geo, push_ind), [-1])
        push_gt = tf.reshape(tf.gather(new_g_geo2, push_ind), [-1])

        pull_loss = tf.cond(tf.shape(pull_feat)[0] > 0,
                            lambda : tf.reduce_mean(tf.abs(pull_feat - pull_gt)),
                            lambda : tf.constant(0.0))
        push_loss = tf.cond(tf.shape(push_feat)[0] > 0,
                            #lambda: tf.reduce_mean(tf.exp((-1. / 1) * tf.abs(push_feat - push_gt))),
                            lambda : tf.reduce_mean(tf.maximum(1.-tf.abs(push_feat - push_gt), 0.)),
                            lambda : tf.constant(0.0))

        slim.losses.add_loss(pull_loss)
        slim.losses.add_loss(push_loss)
        return pull_loss, push_loss


def accurate_compute(pred, label, valid_mask, batch_size):
    '''
    :param pred: [batch_size, x, x, x ....]
    :param label: [batch_size, x, x, x, ...] (must be equal to pred)
    :param valid_mask: [batch_size, x, x, x, ...] (must be equal to pred)
    :param batch_size: int
    :return: accurate_mean: scalar
    '''
    pred = tf.reshape(pred, [batch_size, -1])
    label = tf.reshape(label, [batch_size, -1])
    valid_mask = tf.reshape(valid_mask, [batch_size, -1])

    valid_mask_int = tf.cast(valid_mask, tf.int32)
    valid_mask_float = tf.cast(valid_mask, tf.float32)

    valid_num = tf.reduce_sum(valid_mask_float, axis=1)

    equal_mask = tf.equal(tf.cast(pred,tf.int64), tf.cast(label,tf.int64))
    equal_mask_valid = tf.logical_and(equal_mask, valid_mask)
    equal_mask_valid_float = tf.cast(equal_mask_valid, tf.float32)

    accurate = tf.reduce_sum(equal_mask_valid_float, axis=1)
    accurate = tf.div(accurate, valid_num)
    accurate = tf.where(tf.is_nan(accurate), tf.zeros_like(accurate), accurate)

    batch_div = tf.reduce_sum(tf.cast(tf.not_equal(valid_num, 0), tf.float32))

    accurate_mean = tf.reduce_sum(accurate) / batch_div
    accurate_mean = tf.where(tf.is_nan(accurate_mean), tf.zeros_like(accurate_mean), accurate_mean)

    return accurate_mean


def recall_precision_compute(score, label, match_th, valid_mask):
    # compute recall and precision during training, only for binary classification cases
    score = tf.reshape(score, [-1])
    label = tf.reshape(label, [-1])
    pred = tf.cast(tf.greater_equal(score, match_th), tf.int32)
    vmask = tf.reshape(valid_mask, [-1])

    tp_logical = tf.logical_and(tf.logical_and(tf.equal(pred, 1), tf.equal(label, 1)), vmask)
    fp_logical = tf.logical_and(tf.logical_and(tf.equal(pred, 1), tf.equal(label, 0)), vmask)
    fn_logical = tf.logical_and(tf.logical_and(tf.equal(pred, 0), tf.equal(label, 1)), vmask)

    tp = tf.reduce_sum(tf.cast(tp_logical, tf.float32))
    fp = tf.reduce_sum(tf.cast(fp_logical, tf.float32))
    fn = tf.reduce_sum(tf.cast(fn_logical, tf.float32))

    recall = tf.cond(tf.not_equal(tp + fn, 0),
                     lambda : tf.div(tp, tp + fn),
                     lambda : tf.constant(-1.0))

    precision = tf.cond(tf.not_equal(tp + fp, 0),
                     lambda: tf.div(tp, tp + fp),
                     lambda: tf.constant(-1.0))
    return recall, precision


def sample_instance(logits, labels, valid_mask, sample_num, batch_size, redu=False):
    ret_list_logits = []
    ret_list_labels = []

    for i in range(batch_size):
        logits_i = logits[i, :, :]
        labels_i = labels[i, :]
        mask_i = valid_mask[i, :]

        inds = tf.where(mask_i)
        inds_shuffle = tf.random_shuffle(inds)

        num_i = sample_num[i]
        num_i = tf.minimum(tf.cast(num_i, tf.int32), tf.shape(inds_shuffle)[0])
        size_i = tf.stack((num_i, 1))
        size_i = tf.cast(size_i, tf.int32)

        inds_sample = tf.slice(inds_shuffle, [0, 0], size_i)

        logits_gather = tf.gather_nd(logits_i, inds_sample)
        labels_gather = tf.gather_nd(labels_i, inds_sample)

        pad_num = tf.cast(sample_num[i], tf.int32) - num_i
        padding_b = tf.expand_dims(tf.stack((0, pad_num), axis=0), axis=0)
        padding_a = tf.concat((padding_b, [[0, 0]]), axis=0)
        logits_gather = tf.pad(logits_gather, padding_a)
        labels_gather = tf.pad(labels_gather, padding_b)

        if not redu:
            logits_gather = tf.expand_dims(logits_gather, axis=0)
            labels_gather = tf.expand_dims(labels_gather, axis=0)

        ret_list_logits.append(logits_gather)
        ret_list_labels.append(labels_gather)
    if not redu:
        return ret_list_logits, ret_list_labels
    else:
        return tf.concat(ret_list_logits, axis=0), tf.concat(ret_list_labels, axis=0)


def sample_cls_batch(tensor_a, tensor_b, pmask, nmask, pos_num, neg_num, batch_size):
    '''
    :param tensor_a: [batch_size, proposal_num, 2]
    :param tensor_b: [batch_size, proposal_num]
    :param pmask: [batch_size, proposal_num]
    :param nmask: [batch_size, proposal_num]
    :param pos_num: [batch_size]
    :param neg_num: [batch_size]
    :param batch_size: int
    :return: batch_a: [batch_size, mini_batch_size, 2]
    :return: batch_b: [batch_size, mini_batch_size]
    '''
    plist_a, plist_b = sample_instance(tensor_a, tensor_b, pmask, pos_num, batch_size)
    nlist_a, nlist_b = sample_instance(tensor_a, tensor_b, nmask, neg_num, batch_size)

    alist = []
    blist = []

    for i in range(batch_size):
        alist.append(tf.concat((plist_a[i], nlist_a[i]), axis=1))
        blist.append(tf.concat((plist_b[i], nlist_b[i]), axis=1))

    batch_a = tf.concat(alist, axis=0)
    batch_b = tf.concat(blist, axis=0)

    return batch_a, batch_b


def OHEM_select(logits, label, pmask, nmask, batch_size, class_num, sample_size):
    mask = tf.logical_or(pmask, nmask)
    logits_list = []
    label_list = []
    for i in range(batch_size):
        logits_i = logits[i, :, :]
        label_i = label[i, :]
        mask_i = mask[i, :]

        inds_i = tf.where(mask_i)
        v_logits_i = tf.reshape(tf.gather(logits_i, inds_i), [-1, class_num])
        v_labels_i = tf.reshape(tf.gather(label_i, inds_i), [-1])

        loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=v_logits_i, labels=v_labels_i)
        k = tf.minimum(tf.shape(v_logits_i)[0], sample_size)
        _, top_k_inds = tf.nn.top_k(loss_i, k=k)

        top_k_logits = tf.reshape(tf.gather(v_logits_i, top_k_inds), [-1, class_num])
        top_k_label = tf.reshape(tf.gather(v_labels_i, top_k_inds), [-1])
        logits_list.append(top_k_logits)
        label_list.append(top_k_label)
    return tf.concat(logits_list, axis=0), tf.concat(label_list, axis=0)

def hard_negative_select(gscores, logits, pos_threshold, class_num, batch_size, K):

    hard_neg_logits = []

    for i in range(batch_size):
        l = tf.logical_and(tf.less_equal(gscores[i,:], pos_threshold - 0.05), tf.greater_equal(gscores[i,:], 0.1))
        neg_scores = tf.cast(l, tf.float32) * gscores
        _, top_k_indices = tf.nn.top_k(neg_scores, k=K)
        hard_neg_logit_i = tf.reshape(tf.gather(logits[i,:], top_k_indices), [-1, class_num])
        hard_neg_logits.append(hard_neg_logit_i)
    return tf.concat(hard_neg_logits, axis=0)







