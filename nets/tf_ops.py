import tensorflow as tf
import numpy as np
import math



def anchor_gener(img_shape, feat_shape, sizes, ratios, num_anchors, offset=0.5):
    '''
    :param img_shape: tuple, (img_height, img_width)
    :param feat_shape: tuple, (feat_height, feat_width)
    :param sizes: list
    :param ratios: list
    :param num_anchors: int
    :param offset: float
    :return:
    '''
    dtype = np.float32
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) / feat_shape[0]
    x = (x.astype(dtype) + offset) / feat_shape[1]

    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    h = np.zeros((num_anchors,), dtype=dtype)
    w = np.zeros((num_anchors,), dtype=dtype)

    for i, s in enumerate(sizes):
        for j, r in enumerate(ratios):
            h[i * len(ratios) + j] = (s * math.sqrt(r)) /  img_shape[0]
            w[i * len(ratios) + j] = (s / math.sqrt(r)) /  img_shape[1]
    return (y, x, h, w)

def anchor_tensor_gener(anchors):
    '''
    :param anchors: y, x: [feat_shape[0], feat_shape[1], h, w: [num_anchors]
    :return: anchor_tensor: [feat_shape[0], feat_shape[1], num_anchors, 4], N = feat_shape[0] * feat_shape[1] * num_anchors
    '''
    y, x, h, w = anchors
    anchor_tensor = np.zeros([y.shape[0],y.shape[1],h.shape[0],4], dtype=np.float32)
    for i in range(anchor_tensor.shape[0]):
        for j in range(anchor_tensor.shape[1]):
            for k in range(anchor_tensor.shape[2]):
                anchor_tensor[i,j,k,0] = x[i,j]
                anchor_tensor[i,j,k,1] = y[i,j]
                anchor_tensor[i,j,k,2] = w[k]
                anchor_tensor[i,j,k,3] = h[k]
    return anchor_tensor

def anchor_gener_by_size(img_shape, feat_shape, sizes, num_anchors, offset=0.5, dtype=np.float32):
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) / feat_shape[0]
    x = (x.astype(dtype) + offset) / feat_shape[1]

    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    h = np.zeros((num_anchors,), dtype=dtype)
    w = np.zeros((num_anchors,), dtype=dtype)

    for i, s in enumerate(sizes):
        h[i] = s[0] / img_shape[0]
        w[i] = s[1] / img_shape[1]
    return (y, x, h, w)

def get_anchor_box(anchor_tensor):
    fanchor = tf.reshape(anchor_tensor, [-1, 4])  # [N ,4]

    anchor_x = fanchor[:, 0]
    anchor_y = fanchor[:, 1]
    anchor_w = fanchor[:, 2]
    anchor_h = fanchor[:, 3]

    anchor_ymin = anchor_y - anchor_h * 0.5
    anchor_xmin = anchor_x - anchor_w * 0.5
    anchor_ymax = anchor_y + anchor_h * 0.5
    anchor_xmax = anchor_x + anchor_w * 0.5

    anchor_ymin = tf.maximum(anchor_ymin, 0.)
    anchor_xmin = tf.maximum(anchor_xmin, 0.)
    anchor_ymax = tf.minimum(anchor_ymax, 1.)
    anchor_xmax = tf.minimum(anchor_xmax, 1.)

    ret = tf.stack((anchor_ymin, anchor_xmin, anchor_ymax, anchor_xmax), axis=-1)

    ret = tf.reshape(ret, tf.shape(anchor_tensor))

    return ret

def roi_decode(roi_loc, anchor_tensor, decode=True):
    '''
    :param roi_loc: [batch_size, N, 4], N = feat_shape[0] * feat_shape[1] * anchor_num
    :param anchor_tensor: [feat_shape[0], feat_shape[1], anchor_num, 4], (x,y,w,h)
    :param decode: bool
    :return: [batch_size, N, 4]
    '''
    fanchor = tf.reshape(anchor_tensor, [1,-1,4]) #[1, N ,4]

    anchor_x = fanchor[:,:,0]
    anchor_y = fanchor[:,:,1]
    anchor_w = fanchor[:,:,2]
    anchor_h = fanchor[:,:,3]

    anchor_ymin = anchor_y - anchor_h * 0.5
    anchor_xmin = anchor_x - anchor_w * 0.5
    anchor_ymax = anchor_y + anchor_h * 0.5
    anchor_xmax = anchor_x + anchor_w * 0.5

    anchor_ymin = tf.maximum(anchor_ymin, 0.)
    anchor_xmin = tf.maximum(anchor_xmin, 0.)
    anchor_ymax = tf.minimum(anchor_ymax, 1.)
    anchor_xmax = tf.minimum(anchor_xmax, 1.)

    anchor_y = (anchor_ymax + anchor_ymin) * 0.5
    anchor_x = (anchor_xmax + anchor_xmin) * 0.5
    anchor_h = anchor_ymax - anchor_ymin
    anchor_w = anchor_xmax - anchor_xmin


    cx = roi_loc[:, :, 0] * anchor_w + anchor_x # [batch_num, N]
    cy = roi_loc[:, :, 1] * anchor_h + anchor_y # [batch_num, N]
    w = tf.exp(roi_loc[:, :, 2]) * anchor_w # [batch_num, N]
    h = tf.exp(roi_loc[:, :, 3]) * anchor_h # [batch_num, N]

    if decode:
        ymin = cy - 0.5 * h
        xmin = cx - 0.5 * w
        ymax = cy + 0.5 * h
        xmax = cx + 0.5 * w
    else:
        ymin = anchor_ymin
        xmin = anchor_xmin
        ymax = anchor_ymax
        xmax = anchor_xmax

    ymin = tf.maximum(ymin, 0.)
    xmin = tf.maximum(xmin, 0.)
    ymax = tf.minimum(ymax, 1.)
    xmax = tf.minimum(xmax, 1.)

    rois = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    return rois

def bboxes_decode(rois, rois_loc, clip=True):
    '''
    :param rois: [batch_size, num, 4]
    :param rois_loc: [batch_size, num, 4]
    :param clip: bool
    :return: [batch_size, num, 4]
    '''
    rois_ymin = rois[:,:,0] #[batch_size, num]
    rois_xmin = rois[:,:,1] #[batch_size, num]
    rois_ymax = rois[:,:,2] #[batch_size, num]
    rois_xmax = rois[:,:,3] #[batch_size, num]

    rois_x = (rois_xmin + rois_xmax) / 2. #[batch_size, num]
    rois_y = (rois_ymin + rois_ymax) / 2. #[batch_size, num]
    rois_w = rois_xmax - rois_xmin #[batch_size, num]
    rois_h = rois_ymax - rois_ymin #[batch_size, num]

    loc_x = rois_loc[:,:,0] #[batch_size, num]
    loc_y = rois_loc[:,:,1] #[batch_size, num]
    loc_w = rois_loc[:,:,2] #[batch_size, num]
    loc_h = rois_loc[:,:,3] #[batch_size, num]

    _x = loc_x * rois_w + rois_x #[batch_size, num]
    _y = loc_y * rois_h + rois_y #[batch_size, num]
    _w = tf.exp(loc_w) * rois_w #[batch_size, num]
    _h = tf.exp(loc_h) * rois_h #[batch_size, num]

    _ymin = _y - _h / 2. #[batch_size, num]
    _xmin = _x - _w / 2. #[batch_size, num]
    _ymax = _y + _h / 2. #[batch_size, num]
    _xmax = _x + _w / 2. #[batch_size, num]

    if clip:
        _ymin = tf.maximum(_ymin, 0.)
        _xmin = tf.maximum(_xmin, 0.)
        _ymax = tf.minimum(_ymax, 1.)
        _xmax = tf.minimum(_xmax, 1.)

    bboxes = tf.stack([_ymin,_xmin,_ymax,_xmax],axis=-1) #[batch_size, num, 4]
    return bboxes

def roi_select(roi_score, roi_loc, anchor_tensor, batch_size, proposal_num, select_nms_th,with_score=False):
    '''
    :param roi_score: [batch_size, N]
    :param roi_loc: [batch_size, N, 4]
    :param anchor_tensor: [batch_size, N, 4]
    :param batch_size: int
    :param proposal_num: int
    :param select_nms_th: float
    :param with_score: bool
    :return:
    '''
    rois = roi_decode(roi_loc, anchor_tensor, decode=True)  # [batch_size, N, 4]
    _proposal_num = proposal_num
    rois_list = []
    score_list = []
    for i in range(batch_size):
        roi_score_i = roi_score[i,:]
        rois_i = rois[i,:,:]
        keep_num = tf.minimum(_proposal_num, tf.shape(roi_score_i)[0])
        selected_indices = tf.image.non_max_suppression(rois_i,roi_score_i,keep_num, select_nms_th)
        pad_num = keep_num - tf.shape(selected_indices)[0]
        selected_indices = tf.pad(selected_indices,[[0,pad_num]])
        selected_rois = tf.gather(rois_i, selected_indices)

        selected_score_i = tf.gather(roi_score_i, selected_indices)

        rois_list.append(tf.expand_dims(selected_rois,axis=0))
        score_list.append(tf.expand_dims(selected_score_i,axis=0))


    rois_selected = tf.concat(rois_list, axis=0)
    score_selected = tf.concat(score_list, axis=0)

    top_k_score, top_k_indices = tf.nn.top_k(score_selected, k=proposal_num) #[batch_size, proposal_num] for both
    batch_ind = tf.constant([[i] * proposal_num for i in range(batch_size)])
    gind = tf.stack([batch_ind, top_k_indices], axis=-1)
    gind = tf.reshape(gind, [-1, 2])
    rois_selected = tf.gather_nd(rois_selected, gind)

    rois_selected = tf.reshape(rois_selected, [batch_size, proposal_num, 4])

    if not with_score:
        return rois_selected #[batch_size, proposal_num, 4]
    else:
        return rois_selected, top_k_score

def bboxes_encode(anchors, bboxes, labels, up_th, low_th, net_shape, num_classes):
    dtype = tf.float32
    box_ymin = bboxes[:, 0]
    box_xmin = bboxes[:, 1]
    box_ymax = bboxes[:, 2]
    box_xmax = bboxes[:, 3]

    box_ymin = tf.maximum(box_ymin, 0.)
    box_xmin = tf.maximum(box_xmin, 0.)
    box_ymax = tf.minimum(box_ymax, 1.)
    box_xmax = tf.minimum(box_xmax, 1.)

    bboxes = tf.stack((box_ymin, box_xmin, box_ymax, box_xmax), axis=1)

    yref, xref, href, wref = anchors
    ymin = yref - href / 2.
    xmin = xref - wref / 2.
    ymax = yref + href / 2.
    xmax = xref + wref / 2.

    ymin = tf.maximum(ymin, 0.)
    xmin = tf.maximum(xmin, 0.)
    ymax = tf.minimum(ymax, 1.)
    xmax = tf.minimum(xmax, 1.)

    vol_anchors = (xmax - xmin) * (ymax - ymin)

    # Initialize tensors...
    shape = (yref.shape[0], yref.shape[1], href.size)
    feat_labels = tf.zeros(shape, dtype=tf.int64)
    feat_scores = tf.zeros(shape, dtype=dtype)

    feat_ymin = tf.zeros(shape, dtype=dtype)
    feat_xmin = tf.zeros(shape, dtype=dtype)
    feat_ymax = tf.ones(shape, dtype=dtype)
    feat_xmax = tf.ones(shape, dtype=dtype)

    def jaccard_with_anchors(bbox):
        """Compute jaccard score between a box and the anchors.
        """
        int_ymin = tf.maximum(ymin, bbox[0])
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])

        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        union_vol = vol_anchors - inter_vol \
                    + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        jaccard = tf.div(inter_vol, union_vol)
        return jaccard

    def scale_res(bbox):
        h = bbox[2] - bbox[0]
        w = bbox[3] - bbox[1]
        h = h * net_shape[0]
        w = w * net_shape[1]
        scale = tf.sqrt(h*w)
        valid_mask = tf.logical_and(tf.greater(scale, low_th), tf.less(scale, up_th))
        valid_mask = tf.reshape(valid_mask, [-1, 1])
        return valid_mask


    def condition(i, feat_labels, feat_scores,
                  feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        """Condition: check label index.
        """
        r = tf.less(i, tf.shape(labels))
        return r[0]

    def body(i, feat_labels, feat_scores,
             feat_ymin, feat_xmin, feat_ymax, feat_xmax):

        # Jaccard score.
        label = labels[i]
        bbox = bboxes[i]
        jaccard = jaccard_with_anchors(bbox)
        scale_valid = scale_res(bbox)
        # Mask: check threshold + scores + no annotations + num_classes.
        mask = tf.greater(jaccard, feat_scores)
        mask = tf.logical_and(scale_valid, mask)
        # mask = tf.logical_and(mask, tf.greater(jaccard, matching_threshold))
        mask = tf.logical_and(mask, label < num_classes)
        imask = tf.cast(mask, tf.int64)
        fmask = tf.cast(mask, dtype)
        # Update values using mask.
        feat_labels = imask * label + (1 - imask) * feat_labels
        feat_scores = tf.where(mask, jaccard, feat_scores)

        feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
        feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
        feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
        feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax


        return [i + 1, feat_labels, feat_scores,
                feat_ymin, feat_xmin, feat_ymax, feat_xmax]

    # Main loop definition.
    i = 0
    [i, feat_labels, feat_scores,
     feat_ymin, feat_xmin,
     feat_ymax, feat_xmax] = tf.while_loop(condition, body,
                                           [i, feat_labels, feat_scores,
                                            feat_ymin, feat_xmin,
                                            feat_ymax, feat_xmax])
    # Transform to center / size.
    feat_cy = (feat_ymax + feat_ymin) / 2.
    feat_cx = (feat_xmax + feat_xmin) / 2.
    feat_h = feat_ymax - feat_ymin
    feat_w = feat_xmax - feat_xmin
    # Encode features.
    feat_cy = (feat_cy - yref) / href
    feat_cx = (feat_cx - xref) / wref
    feat_h = tf.log(feat_h / href)
    feat_w = tf.log(feat_w / wref)
    #  x / y / w / h
    feat_localizations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
    feat_localizations = tf.where(tf.is_nan(feat_localizations), tf.zeros_like(feat_localizations), feat_localizations)
    return feat_labels, feat_localizations, feat_scores

def roi_encode(rois,bboxes,labels,up_th, low_th, net_shape, num_classes, max_bbox_num):
    '''
    :param rois: [batch_size, N, 4]
    :param bboxes: [batch_size, max_bboxes_num, 4]
    :param labels:  [batch_size, max_bboxes_num]
    :param up_th: float
    :param low_th: float
    :param net_shape: tuple
    :param num_classes: int
    :param max_bbox_num: int
    :return:
    '''
    dtype = tf.float32
    ymin = rois[:, :, 0]  # [batch_size, N]
    xmin = rois[:, :, 1]  # [batch_size, N]
    ymax = rois[:, :, 2]  # [batch_size, N]
    xmax = rois[:, :, 3]  # [batch_size, N]
    vol_anchors = (xmax - xmin) * (ymax - ymin)  # [batch_size, N]

    # Initialize tensors...
    shape = ymin.get_shape()
    feat_scores = tf.zeros(shape, dtype=dtype)  # [batch_size, N]
    feat_scores2 = tf.zeros(shape, dtype=dtype)  # [batch_size, N]

    feat_labels = tf.zeros(shape, dtype=tf.int64)
    feat_labels2 = tf.zeros(shape, dtype=tf.int64)

    feat_ymin = tf.zeros(shape, dtype=dtype)  # [batch_size, N]
    feat_xmin = tf.zeros(shape, dtype=dtype)  # [batch_size, N]
    feat_ymax = tf.ones(shape, dtype=dtype)  # [batch_size, N]
    feat_xmax = tf.ones(shape, dtype=dtype)  # [batch_size, N]

    feat_ymin2 = tf.zeros(shape, dtype=dtype)  # [batch_size, N]
    feat_xmin2 = tf.zeros(shape, dtype=dtype)  # [batch_size, N]
    feat_ymax2 = tf.ones(shape, dtype=dtype)  # [batch_size, N]
    feat_xmax2 = tf.ones(shape, dtype=dtype)  # [batch_size, N]


    def jaccard_with_anchors(bbox):
        """Compute jaccard score between a box and the anchors.
        """
        b_ymin = tf.reshape(bbox[:, 0], [-1, 1]) #[batch_size, 1]
        b_xmin = tf.reshape(bbox[:, 1], [-1, 1]) #[batch_size, 1]
        b_ymax = tf.reshape(bbox[:, 2], [-1, 1]) #[batch_size, 1]
        b_xmax = tf.reshape(bbox[:, 3], [-1, 1]) #[batch_size, 1]

        int_ymin = tf.maximum(ymin, b_ymin) #[batch_size, N]
        int_xmin = tf.maximum(xmin, b_xmin) #[batch_size, N]
        int_ymax = tf.minimum(ymax, b_ymax) #[batch_size, N]
        int_xmax = tf.minimum(xmax, b_xmax) #[batch_size, N]

        h = tf.maximum(int_ymax - int_ymin, 0.) #[batch_size, N]
        w = tf.maximum(int_xmax - int_xmin, 0.) #[batch_size, N]
        # Volumes.
        inter_vol = h * w #[batch_size, N]
        union_vol = vol_anchors - inter_vol + (b_ymax - b_ymin) * (b_xmax - b_xmin) #[batch_size, N]
        jaccard = tf.div(inter_vol, union_vol) #[batch_size, N]
        return jaccard #[batch_size, N]

    def scale_res(bbox):
        h = bbox[:,2] - bbox[:,0]
        w = bbox[:,3] - bbox[:,1]
        h = h * net_shape[0]
        w = w * net_shape[1]
        scale = tf.sqrt(h*w)
        valid_mask = tf.logical_and(tf.greater(scale, low_th), tf.less(scale, up_th))
        valid_mask = tf.reshape(valid_mask, [-1, 1])
        return valid_mask


    def condition(i, feat_scores,feat_2_scores, feat_labels, feat_labels2, feat_ymin2,feat_xmin2,feat_ymax2, feat_xmax2, feat_ymin, feat_xmin,feat_ymax,feat_xmax):
        r = tf.less(i, max_bbox_num)
        return r


    def body(i, feat_scores,feat_2_scores, feat_labels, feat_labels2, feat_ymin2,feat_xmin2,feat_ymax2, feat_xmax2, feat_ymin, feat_xmin,feat_ymax,feat_xmax):
        label = labels[:, i]  # [batch_size]
        label = tf.reshape(label, [-1, 1])  # [batch_size, 1]
        bbox = bboxes[:, i, :]  # [batch_size, 4]
        jaccard = jaccard_with_anchors(bbox)  # [batch_size, N]

        scale_valid = scale_res(bbox)

        mask1 = tf.greater(jaccard, feat_scores)  # [batch_size, N]
        mask1 = tf.logical_and(scale_valid, mask1)
        mask1 = tf.logical_and(mask1, label < num_classes)  # [batch_size, N]

        mask2 = tf.logical_and(tf.greater(jaccard, feat_2_scores), tf.less(jaccard, feat_scores))
        mask2 = tf.logical_and(scale_valid, mask2)
        mask2 = tf.logical_and(mask2, label < num_classes)

        imask1 = tf.cast(mask1, tf.int64)  # [batch_size, N]
        fmask1 = tf.cast(mask1, dtype)  # [batch_size, N]
        imask2 = tf.cast(mask2, tf.int64)  # [batch_size, N]
        fmask2 = tf.cast(mask2, dtype)  # [batch_size, N]

        # Update values using mask.
        feat_2_scores = fmask1 * feat_scores + fmask2 * jaccard + (1.-fmask1) * (1.-fmask2) * feat_2_scores
        feat_scores = tf.where(mask1, jaccard, feat_scores)  # [batch_size, N]

        feat_labels2 = imask1 * feat_labels + imask2 * label + (1-imask1)*(1-imask2) * feat_labels2
        feat_labels = imask1 * label + (1-imask1) * feat_labels

        feat_ymin2 = fmask1 * feat_ymin + fmask2 * tf.reshape(bbox[:, 0], [-1, 1]) + (1.-fmask1) * (1.-fmask2) * feat_ymin2  # [batch_size, N]
        feat_xmin2 = fmask1 * feat_xmin + fmask2 * tf.reshape(bbox[:, 1], [-1, 1]) + (1.-fmask1) * (1.-fmask2) * feat_xmin2  # [batch_size, N]
        feat_ymax2 = fmask1 * feat_ymax + fmask2 * tf.reshape(bbox[:, 2], [-1, 1]) + (1.-fmask1) * (1.-fmask2) * feat_ymax2  # [batch_size, N]
        feat_xmax2 = fmask1 * feat_xmax + fmask2 * tf.reshape(bbox[:, 3], [-1, 1]) + (1.-fmask1) * (1.-fmask2) * feat_xmax2  # [batch_size, N]

        feat_ymin = fmask1 * tf.reshape(bbox[:, 0], [-1, 1]) + (1 - fmask1) * feat_ymin  # [batch_size, N]
        feat_xmin = fmask1 * tf.reshape(bbox[:, 1], [-1, 1]) + (1 - fmask1) * feat_xmin  # [batch_size, N]
        feat_ymax = fmask1 * tf.reshape(bbox[:, 2], [-1, 1]) + (1 - fmask1) * feat_ymax  # [batch_size, N]
        feat_xmax = fmask1 * tf.reshape(bbox[:, 3], [-1, 1]) + (1 - fmask1) * feat_xmax  # [batch_size, N]

        return [i+1, feat_scores,feat_2_scores, feat_labels, feat_labels2, feat_ymin2,feat_xmin2,feat_ymax2, feat_xmax2, feat_ymin, feat_xmin,feat_ymax,feat_xmax]

    i = 0
    [i, feat_scores,feat_scores2, feat_labels, feat_labels2,
     feat_ymin2,feat_xmin2,feat_ymax2, feat_xmax2,
     feat_ymin, feat_xmin,feat_ymax,feat_xmax] = tf.while_loop(condition, body,
                                                            [i, feat_scores,feat_scores2, feat_labels, feat_labels2,
                                                            feat_ymin2,feat_xmin2,feat_ymax2, feat_xmax2,
                                                            feat_ymin, feat_xmin,feat_ymax,feat_xmax])

    feat_cy2 = (feat_ymax2 + feat_ymin2) / 2.  # [batch_size, N]
    feat_cx2 = (feat_xmax2 + feat_xmin2) / 2.  # [batch_size, N]
    feat_h2 = feat_ymax2 - feat_ymin2  # [batch_size, N]
    feat_w2 = feat_xmax2 - feat_xmin2  # [batch_size, N]

    feat_cy = (feat_ymax + feat_ymin) / 2.  # [batch_size, N]
    feat_cx = (feat_xmax + feat_xmin) / 2.  # [batch_size, N]
    feat_h = feat_ymax - feat_ymin  # [batch_size, N]
    feat_w = feat_xmax - feat_xmin  # [batch_size, N]

    # Encode features.
    yref = (ymin + ymax) / 2.  # [batch_size, N]
    xref = (xmin + xmax) / 2.  # [batch_size, N]
    href = ymax - ymin  # [batch_size, N]
    wref = xmax - xmin  # [batch_size, N]

    feat_cy = (feat_cy - yref) / href  # [batch_size, N]
    feat_cx = (feat_cx - xref) / wref  # [batch_size, N]
    feat_h = tf.log(feat_h / href)  # [batch_size, N]
    feat_w = tf.log(feat_w / wref)  # [batch_size, N]

    feat_cy2 = (feat_cy2 - yref) / href  # [batch_size, N]
    feat_cx2 = (feat_cx2 - xref) / wref  # [batch_size, N]
    feat_h2 = tf.log(feat_h2 / href)  # [batch_size, N]
    feat_w2 = tf.log(feat_w2 / wref)  # [batch_size, N]

    feat_location = tf.stack([feat_ymin, feat_xmin, feat_ymax, feat_xmax], axis=-1)
    feat_location2 = tf.stack([feat_ymin2, feat_xmin2, feat_ymax2, feat_xmax2], axis=-1)


    feat_delta = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)  # [batch_size, N, 4]
    feat_delta2 = tf.stack([feat_cx2, feat_cy2, feat_w2, feat_h2], axis=-1)  # [batch_size, N, 4]

    feat_delta = tf.where(tf.is_nan(feat_delta), tf.zeros_like(feat_delta), feat_delta)
    feat_delta2 = tf.where(tf.is_nan(feat_delta2), tf.zeros_like(feat_delta2), feat_delta2)

    return feat_labels, feat_delta, feat_scores, feat_location, feat_labels2, feat_delta2, feat_scores2, feat_location2

def apply_random_distortion(boxes, batch_size, N, r_xy, r_wh):
    rand_xy = tf.random_uniform([batch_size, N, 2], minval=-1 * r_xy, maxval=r_xy, dtype=tf.float32)
    rand_wh = tf.random_uniform([batch_size, N, 2], minval=-1 * r_wh, maxval=r_wh, dtype=tf.float32)
    rand_loc = tf.concat((rand_xy, rand_wh), axis=2)
    d_boxes = bboxes_decode(boxes, rand_loc)
    return d_boxes




