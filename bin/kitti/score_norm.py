import os
import sys
import argparse
from pathlib import Path
sys.path.append(os.path.abspath(Path(__file__).parent.parent))

import numpy as np


def score_norm(scores):
    scores = np.array(scores) * 100000.
    inds = np.arange(scores.shape[0])

    e_scores = np.expand_dims(scores, axis=1)
    e_inds = np.expand_dims(inds, axis=1)

    c_score = np.concatenate((e_scores, e_inds), axis=1)
    np.random.shuffle(c_score)

    s_scores = c_score[:,0]
    s_inds = c_score[:,1].astype(np.int)

    arg_sort_ind = np.argsort(s_scores)
  
    new_scores = s_scores.copy()
    ret_scores = np.zeros_like(scores)

    for i in range(scores.shape[0]):
        new_scores[arg_sort_ind[i]] = float(i)
    for i in range(scores.shape[0]):
        ret_scores[s_inds[i]] = new_scores[i]
    return ret_scores

def run_score_norm(ind_dict, scores):
    normed_scores = score_norm(scores)

    score_dict = {}

    for file_name in ind_dict:
        score_dict[file_name] = []
        for ind in ind_dict[file_name]:
            score_dict[file_name].append(normed_scores[ind])
    return score_dict


def make_record(pred_dict, output_dir):
    for fpred in pred_dict:
        f = open(os.path.join(output_dir, fpred), 'w')
        n = len(pred_dict[fpred])
        if n == 0:
            f.close()
            continue
        for j in range(n):
            f.write('%d %d %d %d %d %.2f\n' % (
                1,
                pred_dict[fpred][j]['box'][1], 
                pred_dict[fpred][j]['box'][0], 
                pred_dict[fpred][j]['box'][3], 
                pred_dict[fpred][j]['box'][2],
                pred_dict[fpred][j]['score']))
        f.close()


def run(pred_dir, output_dir):
    flist = os.listdir(pred_dir)
    ind_dict = {}
    pred_dict = {}
    scores = []
    k = 0
    for fpred in flist:
        ind_dict[fpred] = []
        preds = kitti_eval_fuck.load_pred_raw(os.path.join(pred_dir, fpred))
        for pred in preds:
            scores.append(pred['score'])
            ind_dict[fpred].append(k)
            k += 1
        pred_dict[fpred] = preds
    
    print('Reading detections completed!')
    
    new_score_dict = run_score_norm(ind_dict, scores)
    for fpred in flist:
        for i in range(len(pred_dict[fpred])):
            pred_dict[fpred][i]['score'] = new_score_dict[fpred][i]
    print('Converting scores completed!')
    make_record(pred_dict, output_dir)
    print('Writing results completed!')




    
    
































