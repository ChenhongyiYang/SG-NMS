import numpy as np

def get_iou(tbox, rest_box):
    tbox = np.reshape(tbox, [1, 5])
    rvol = (rest_box[:, 2] - rest_box[:, 0]) * (rest_box[:, 3] - rest_box[:, 1])
    tvol = (tbox[:, 2] - tbox[:, 0]) * (tbox[:, 3] - tbox[:, 1])

    iymin = np.maximum(tbox[:, 0], rest_box[:, 0])
    ixmin = np.maximum(tbox[:, 1], rest_box[:, 1])
    iymax = np.minimum(tbox[:, 2], rest_box[:, 2])
    ixmax = np.minimum(tbox[:, 3], rest_box[:, 3])

    ih = iymax - iymin
    iw = ixmax - ixmin
    ivol = ih * iw

    valid_iou = np.logical_and(ih > 0, iw > 0).astype(np.float)

    iou = ivol / (rvol + tvol - ivol)
    iou = iou * valid_iou
    return iou

def nms_cpu(boxes, scores, ae, mode, Nt, threshold, sigma, ae_dis_th):
    if scores.shape[0] == 0:
        return None, None
    i = 0

    _boxes = np.copy(boxes)
    _scores = np.copy(scores)

    # compute associate embedding
    _ae = np.copy(ae)

    box_ind = np.arange(_boxes.shape[0]).reshape([-1, 1])
    _boxes = np.concatenate((_boxes, box_ind), axis=1)
    while i < _boxes.shape[0] - 1:
        max_pos = np.argmax(_scores[i:]) + i

        tscore = _scores[max_pos]
        tbox = _boxes[max_pos, :].copy()
        tae = _ae[max_pos]

        _scores[max_pos] = _scores[i]
        _scores[i] = tscore

        _boxes[max_pos, :] = _boxes[i, :]
        _boxes[i, :] = tbox

        _ae[max_pos] = _ae[i]
        _ae[i] = tae

        rest_box = _boxes[i+1:, :]
        rest_ae = _ae[i+1:]

        tbox = np.reshape(tbox, [1,5])

        iou = get_iou(tbox, rest_box)
        ae_dis = np.abs(tae-rest_ae)

        # Soft-NMS, linear
        if mode == 0:
            w = (1. - iou) * (np.greater(iou, Nt)).astype(np.float32) + (np.less_equal(iou, Nt)).astype(np.float32)

            # Soft-NMS, gaussian
        elif mode == 1:
            w = np.exp(-(iou * iou) / sigma)

            # Greedy-NMS
        elif mode == 2:
            w = np.less_equal(iou, Nt).astype(np.float32)

            # AE-NMS
        elif mode == 3:
            l1 = np.less_equal(iou, Nt)
            l2 = np.greater(ae_dis, ae_dis_th)
            w = np.logical_or(l1, l2).astype(np.float32)

        elif mode == 4:
            l1 = np.less_equal(iou, Nt)
            l2 = np.greater(ae_dis, ae_dis_th * iou)
            w = np.logical_or(l1, l2).astype(np.float32)

        else:
            l1 = np.less_equal(iou, Nt)
            l2 = np.greater(ae_dis, ae_dis_th * iou ** 2)
            w = np.logical_or(l1, l2).astype(np.float32)

        _scores[i+1:] = _scores[i+1:] * w
        inds = np.where(_scores > threshold)
        _scores = _scores[inds]
        _boxes = _boxes[inds]
        _ae = _ae[inds]
        i += 1

    got_inds = _boxes[:,4].reshape([-1]).astype(np.int)
    ret_scores = scores[got_inds]
    ret_boxes = boxes[got_inds]
    return ret_boxes, ret_scores

def second_nms(boxes_list, scores_list, Nt, threshold):
    use_boxes_list = []
    use_scores_list = []

    for i,scores in enumerate(scores_list):
        if scores.shape[0] != 0:
            use_boxes_list.append(boxes_list[i])
            use_scores_list.append(scores_list[i])

    if len(use_boxes_list) == 0:
        return None, None
    elif len(use_boxes_list) == 1:

        return use_boxes_list[0], use_scores_list[0]

    use_boxes = []
    use_scores = []
    for i in range(len(use_boxes_list)):
        use_boxes_i = use_boxes_list[i].copy() #[n, 4]
        use_scores_i = use_scores_list[i].copy() #[n]
        layer_ind = np.reshape(np.ones_like(use_scores_i) * i, [-1,1])
        use_boxes_i = np.concatenate((use_boxes_i, layer_ind), axis=1) #[n,5]
        use_boxes.append(use_boxes_i)
        use_scores.append(use_scores_i)

    use_boxes = np.concatenate(use_boxes, axis=0)
    use_scores = np.concatenate(use_scores, axis=0)

    _boxes = np.copy(use_boxes)
    _scores = np.copy(use_scores)

    box_ind = np.arange(_boxes.shape[0]).reshape([-1, 1])
    _boxes = np.concatenate((_boxes, box_ind), axis=1)

    i = 0
    while i < _boxes.shape[0] - 1:
        max_pos = np.argmax(_scores[i:]) + i

        tscore = _scores[max_pos]
        tbox = _boxes[max_pos, :].copy()

        _scores[max_pos] = _scores[i]
        _scores[i] = tscore

        _boxes[max_pos, :] = _boxes[i, :]
        _boxes[i, :] = tbox

        # IoU computing
        rest_box = _boxes[i + 1:, :]
        rvol = (rest_box[:, 2] - rest_box[:, 0]) * (rest_box[:, 3] - rest_box[:, 1])

        tbox = np.reshape(tbox, [1, 6])
        tvol = (tbox[:, 2] - tbox[:, 0]) * (tbox[:, 3] - tbox[:, 1])

        iymin = np.maximum(tbox[:, 0], rest_box[:, 0])
        ixmin = np.maximum(tbox[:, 1], rest_box[:, 1])
        iymax = np.minimum(tbox[:, 2], rest_box[:, 2])
        ixmax = np.minimum(tbox[:, 3], rest_box[:, 3])

        ih = iymax - iymin
        iw = ixmax - ixmin
        ivol = ih * iw

        valid_iou = np.logical_and(ih > 0, iw > 0).astype(np.float)

        iou = ivol / (rvol + tvol - ivol)
        iou = iou * valid_iou

        w1 = np.less(iou, Nt).astype(np.float32)
        w2 = np.equal(rest_box[:,4], tbox[0,4])
        w = np.logical_or(w1, w2).astype(np.float)

        _scores[i + 1:] = _scores[i + 1:] * w
        inds = np.where(_scores > threshold)
        _scores = _scores[inds]
        _boxes = _boxes[inds]
        i += 1

    got_inds = _boxes[:, 5].reshape([-1]).astype(np.int)
    ret_scores = use_scores[got_inds]
    ret_boxes = use_boxes[:,:4][got_inds]
    return ret_boxes, ret_scores

def nms_all_class(boxes, scores, classes, aes, class_num, mode, threshold, Nt, sigma, ae_dis, zero_is_back=True):
    if zero_is_back:
        catagories = list(range(1,class_num))
    else:
        catagories = list(range(class_num))
    ret_scores = []
    ret_classes = []
    ret_boxes = []
    for c in catagories:
        cinds = np.where(classes == c)
        c_boxes = boxes[cinds]
        c_scores = scores[cinds]
        c_ae = aes[cinds]

        nms_boxes, nms_scores = nms_cpu(c_boxes, c_scores, c_ae, mode, threshold=threshold, Nt=Nt, sigma=sigma, ae_dis_th=ae_dis)

        if nms_scores is not None:
            nms_classes = np.ones_like(nms_scores) * c
            ret_scores.append(nms_scores)
            ret_boxes.append(nms_boxes)
            ret_classes.append(nms_classes)

    if len(ret_scores) > 0:
        ret_scores = np.concatenate(ret_scores, axis=0)
        ret_classes = np.concatenate(ret_classes, axis=0)
        ret_boxes = np.concatenate(ret_boxes, axis=0)

        ret_scores = [ret_scores.reshape([-1])]
        ret_classes = [ret_classes.reshape([-1])]
        ret_boxes = [ret_boxes.reshape([-1,4])]

    return ret_boxes, ret_scores, ret_classes

def second_nms_all_class(boxes, scores, classes, class_num, Nt, threshold,zero_is_back=True):
    if zero_is_back:
        catagories = list(range(1,class_num))
    else:
        catagories = list(range(class_num))

    ret_scores = []
    ret_boxes = []
    ret_classes = []

    for c in catagories:
        nms_box_list = []
        nms_scores_list = []
        for i in range(len(boxes)):
            cinds = np.where(classes[i] == c)
            c_boxes = boxes[i][cinds]
            c_scores = scores[i][cinds]
            nms_box_list.append(c_boxes)
            nms_scores_list.append(c_scores)

        nms_boxes, nms_scores = second_nms(nms_box_list,nms_scores_list, Nt, threshold)
        if nms_scores is not None:
            nms_classes = np.ones_like(nms_scores) * c
            ret_scores.append(nms_scores)
            ret_boxes.append(nms_boxes)
            ret_classes.append(nms_classes)

    if len(ret_scores) > 0:
        ret_scores = np.concatenate(ret_scores, axis=0)
        ret_classes = np.concatenate(ret_classes, axis=0)
        ret_boxes = np.concatenate(ret_boxes, axis=0)

        ret_scores = [ret_scores.reshape([-1])]
        ret_classes = [ret_classes.reshape([-1])]
        ret_boxes = [ret_boxes.reshape([-1,4])]
    return ret_boxes, ret_scores, ret_classes


































