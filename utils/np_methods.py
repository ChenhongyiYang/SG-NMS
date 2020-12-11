import numpy as np

def get_iou(box1, box2):
    ymin1, xmin1, ymax1, xmax1 = box1
    ymin2, xmin2, ymax2, xmax2 = box2

    iymin = np.maximum(ymin1, ymin2)
    ixmin = np.maximum(xmin1, xmin2)
    iymax = np.minimum(ymax1, ymax2)
    ixmax = np.minimum(xmax1, xmax2)

    h = np.maximum(iymax - iymin, 0.)
    w = np.maximum(ixmax - ixmin, 0.)

    ivol = h * w
    vol1 = (ymax1 - ymin1) * (xmax1 - xmin1)
    vol2 = (ymax2 - ymin2) * (xmax2 - xmin2)

    iou = ivol / (vol1 + vol2 - ivol)
    return iou






























