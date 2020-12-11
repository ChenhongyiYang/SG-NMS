import os
import sys
import argparse
from pathlib import Path
sys.path.append(os.path.abspath(Path(__file__).parent.parent))
import time


from citypersons import cp_preprocess,cp_eval
from inference import inference_nms
from remote.citypersons.configs import cp_config as cp_config

import argparse

parser = argparse.ArgumentParser(description='Tensorflow rfcn_occ testing')
parser.add_argument('--gpu', required=True, type=str, help='gpu indexes')
parser.add_argument('--list', required=False, type=int, default=0, help='whether save raw list')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

test_img_dir = '/path/to/test_img_dir'
out_txt_dir = '/path/to/output_dir'

model = '/path/to/ckpt_dir/model_sample.ckpt-240000'
cp_nets_param = cp_config.cp_test_param

# nms params
Nt_all = 0.45
ae_dis = 1.7

param = {
    'ths':[0.01, 0.01, 0.01],
    'th2':0.01,
    'Nts':[Nt_all, Nt_all, Nt_all],
    'Nt2':0.51,
    'sigma':0.6,
    'ae_dis':[2.9, 3.0, 2.9],
    'mode':2,
    'second_nms':False
}

dash_line = '\n\n' + '=' * 50
print(dash_line)
print('==> GPU index:', args.gpu)
print('==> Test set: %s'%test_img_dir)
print('==> Output dir: %s'%out_txt_dir)
print('==> Model: %s'%model)
print('==> NMS params:')
print(param)
print(dash_line)

def run():

    tester = inference_nms.Tester(cp_nets_param, (cp_config.cp_height, cp_config.cp_width),
                                  cp_preprocess.preprocess_for_eval, model, out_txt_dir)
    if args.list == 1:
        tester.run(test_img_dir,nms_param=param, is_list=True)
    else:
        tester.run(test_img_dir,nms_param=param, is_list=False)
        cp_eval.run_evaluate_when_testing(out_txt_dir, 'val_gt.json')


if __name__ == '__main__':
    #run()
    cp_eval.run_evaluate_when_testing(out_txt_dir, 'val_gt.json')














































