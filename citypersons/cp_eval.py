import os
from citypersons.evalutation.coco import COCO
from citypersons.evalutation.eval_MR_multisetup import COCOeval
from citypersons.evalutation import convert_to_coco
import copy
from tqdm import tqdm
import shutil


def evaluate_result(cocoDt, coco_gt, res_txt, plot, _print=True, ret=False):
    annType = 'bbox'
    if res_txt is not None:
        res_file = open(res_txt, "w")
    else:
        res_file = None
    mrs = []
    fppis = []
    ms = []
    mean_mrs = []

    imgIds = sorted(coco_gt.getImgIds())
    for id_setup in range(0, 4):
        cocoEval = COCOeval(copy.deepcopy(coco_gt), cocoDt, annType,sub_type=0)
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate(id_setup)
        cocoEval.accumulate()
        mean_mr = cocoEval.summarize(id_setup, res_file, _print)
        mean_mrs.append(mean_mr)
        if plot:
            cocoEval.plot(id_setup,mrs,fppis,ms)
    if res_txt is not None:
        res_file.close()
    if ret:
        return mean_mrs

def evaluate_two_subtype(cocoDt, coco_gt, _print):
    annType = 'bbox'
    mean_mrs = []
    imgIds = sorted(coco_gt.getImgIds())
    for id_setup in range(0, 4):
        cocoEval = COCOeval(copy.deepcopy(coco_gt), cocoDt, annType, sub_type=0)
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate(id_setup)
        cocoEval.accumulate()
        mean_mr = cocoEval.summarize(id_setup, None, _print)
        mean_mrs.append(mean_mr)
    if _print:
        print('\n\n')
    for id_setup in range(0, 4):
        cocoEval = COCOeval(copy.deepcopy(coco_gt), cocoDt, annType, sub_type=1)
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate(id_setup)
        cocoEval.accumulate()
        mean_mr = cocoEval.summarize(id_setup, None, _print)
        mean_mrs.append(mean_mr)
    return mean_mrs


def evaluate(det_json, ann_json, res_txt, plot, _print=True, ret=False):
    cocoGt = COCO(ann_json)
    cocoDt = cocoGt.loadRes(det_json)
    return evaluate_result(cocoDt, cocoGt, res_txt, plot, _print, ret)


def run_evaluate(txt_dir, ann_json, res_txt, temp_json='temp.json'):
    convert_to_coco.convert_to_coco(txt_dir, ann_json, temp_json)
    evaluate(temp_json, ann_json, res_txt, True)
    os.remove(temp_json)

def run_evaluate_when_testing(txt_dir, ann_json, temp_json='temp.json', _print=True):
    convert_to_coco.convert_to_coco(txt_dir, ann_json, temp_json)
    cocoGt = COCO(ann_json)
    cocoDt = cocoGt.loadRes(temp_json)
    mean_mrs = evaluate_two_subtype(cocoDt, cocoGt, _print)
    os.remove(temp_json)
    return mean_mrs


def run_multiple_evaluate(pred_dirs, temp_json_dir, ann_json):
    if os.path.isdir(temp_json_dir):
        shutil.rmtree(temp_json_dir)
    os.mkdir(temp_json_dir)

    print('==> Converting detections:')
    temp_jsons = []
    for k in tqdm(range(len(pred_dirs))):
        pred_dir = pred_dirs[k]
        convert_to_coco.convert_to_coco(pred_dir, ann_json, os.path.join(temp_json_dir, '%d.json'%k))
        temp_jsons.append(os.path.join(temp_json_dir, '%d.json'%k))

    cocoGt = COCO(ann_json)
    mrs_all = []
    print('==> Evaluating detections:')
    for k in tqdm(range(len(temp_jsons))):
        temp_json = temp_jsons[k]
        cocoDt = cocoGt.loadRes(temp_json)
        mean_mrs = evaluate_two_subtype(cocoDt, cocoGt, False)
        mrs_all.append(mean_mrs)
    shutil.rmtree(temp_json_dir)
    return mrs_all









if __name__ == '__main__':
    txt_dir = '/Users/yangchenhongyi/Documents/TEMP/cp_result_2'
    ann_json = '/Users/yangchenhongyi/Documents/TEMP/val_gt.json'
    res_txt = '/Users/yangchenhongyi/Documents/TEMP/cp_result.txt'

    run_evaluate(txt_dir, ann_json, res_txt)




