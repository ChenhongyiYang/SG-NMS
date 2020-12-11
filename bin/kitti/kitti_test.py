import os
import sys
import argparse
from pathlib import Path
sys.path.append(os.path.abspath(Path(__file__).parent.parent))


from kitti import preprocess
from inference import inference_nms
from remote.kitti.configs import kitti_config as kitti_config
from kitti import kitti_eval

parser = argparse.ArgumentParser(description='Tensorflow rfcn_occ testing')
parser.add_argument('--gpu', required=True, type=str, help='gpu indexes')
parser.add_argument('--list', required=False, type=int, default=0, help='whether save raw list')
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

k_width, k_height = 1242, 375

kitti_nets_param = kitti_config.inference_net_param



train_image_root= '/path/to/train_image'
train_label_root = '/path/to/train_label'

eval_image_root = '/path/to/eval_image'
eval_label_root = '/path/to/eval_label'


# validation model
model = '/path/to/ckpt_dir/sample.ckpt-80000'

out_txt_dir = '/path/to/output_dir'


Nt_all = 1.2
ae_dis = 1.7
param = {'ths':[0.01, 0.01, 0.01],
         'th2':0.01,
         'Nts':[Nt_all, Nt_all, Nt_all],
         'Nt2': 0.60,
         'sigma':0.05,
         'ae_dis':[ae_dis, ae_dis, ae_dis],
         'mode': 2,
         'second_nms': False
         }

dash_line = '\n\n' + '=' * 50
print(dash_line)
print('==> GPU index:', args.gpu)
print('==> Output dir: %s'%out_txt_dir)
print('==> Model: %s'%model)
print('==> NMS params:')
print(param)
print(dash_line)


def run():
    tester = inference_nms.Tester(
        kitti_nets_param, 
        (k_height, k_width),
        preprocess.preprocess_for_eval, 
        model, 
        out_txt_dir
    )

    if args.list == 1:
        tester.run(eval_image_root ,nms_param=param, is_list=True)
    else:
        tester.run(eval_image_root, nms_param=param, is_list=False)
        kitti_eval.eval(eval_label_root, out_txt_dir, 'Car')


if __name__ == '__main__':
    run()


































