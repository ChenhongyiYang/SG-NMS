import os
import sys
import argparse
from pathlib import Path
sys.path.append(os.path.abspath(Path(__file__).parent.parent))

from train import train
from kitti import preprocess
from remote.kitti.configs import kitti_config
import time

parser = argparse.ArgumentParser(description='Tensorflow rfcn_occ training')
parser.add_argument('--resume', type=int, help='resume from checkpoint')
parser.add_argument('--log', required=True, type=int, help='whether log')
parser.add_argument('--gpu', required=True, type=str, help='gpu inds')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
print('==> GPU index:',args.gpu)

gpus = []
for c in args.gpu:
    try:
        gpus.append(int(c))
    except:
        continue

learning_rate = 0.0001
epoch_num = 300
log_step = 20
save_steps = [60000, 80000]
batch_size = 4
decay_rate = 0.1
decay_steps = [99999999999]

k_width, k_height = 1242, 375


train_log_dir = '/path/to/train_log_dir'
train_model_dir = '/path/to/train_model_dir'
backbone_model = '/path/to/resnet_v1_101.ckpt'
fine_tune_model = '' # '/path/to/fine_tune_model'


kitti_nets_param = kitti_config.train_net_param
kitti_train_param = train.train_params(
        net='rfcn_occ',
        img_shape=(k_height, k_width),
        net_shape=(kitti_nets_param.net_shape[0], kitti_nets_param.net_shape[1]),
        max_box_num=22,
        emergency_save_file='emergency_save.txt',
        train_log_dir=train_log_dir,
        train_model_dir=train_model_dir,
        backbone_model=backbone_model,
        fine_tune_model=fine_tune_model,
        learning_rate=learning_rate,
        epoch_num=epoch_num,
        decay_steps=decay_steps,
        log_step=log_step,
        decay_rate=decay_rate,
        save_steps=save_steps,
        batch_size=batch_size
)


train_image_root= '/path/to/train_image'
train_label_root = '/path/to/train_label'

eval_image_root = '/path/to/eval_image'
eval_label_root = '/path/to/eval_label'


def train_kitti(continue_train, is_log):
    trainer = train.Trainer(
        kitti_train_param, 
        kitti_nets_param, 
        preprocess.preproces_for_train, 
        train_image_root, 
        train_label_root, 
        gpus, 
        data='kitti'
    )

    log_filename = 'sg_nms_kitti'
    trainer.train(continue_train, log_filename, is_log)


if __name__ == '__main__':
    if args.resume:
        if args.log:
            train_kitti(continue_train=True, is_log=True)
        else:
            train_kitti(continue_train=True, is_log=False)
    else:
        if args.log:
            train_kitti(continue_train=False, is_log=True)
        else:
            train_kitti(continue_train=False, is_log=False)









