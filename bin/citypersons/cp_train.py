import sys
sys.path.append('/home/grad3/hongyi/AE-NMS')
from train import train
from remote.citypersons import cp_preprocess
from remote.citypersons.configs import config4 as cp_config
import time

import os
import argparse


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


learning_rate=0.00001
epoch_num=999999999
log_step=20
decay_rate=0.1
decay_steps = [999999999999]
save_steps = [200000, 240000]
batch_size=4


cp_nets_param = cp_config.cp_train_param

train_log_dir = '/path/to/train_log_dir'
train_model_dir = '/path/to/train_model_dir'
backbone_model = '/path/to/resnet_v1_101.ckpt'
fine_tune_model = '' # '/path/to/fine_tune_model'


cp_train_param = train.train_params(
        net='rfcn_occ',
        img_shape=(cp_nets_param.img_shape[0], cp_nets_param.img_shape[1]),
        net_shape=(cp_nets_param.net_shape[0], cp_nets_param.net_shape[1]),
        max_box_num=cp_nets_param.max_bbox_num,
        emergency_save_file='emergency_save.txt',
        train_log_dir=train_log_dir,
        train_model_dir=train_model_dir,
        backbone_model=backbone_model,
        fine_tune_model=fine_tune_model,
        learning_rate=learning_rate,
        epoch_num=epoch_num,
        log_step=log_step,
        decay_rate=decay_rate,
        decay_steps=decay_steps,
        save_steps=save_steps,
        batch_size=batch_size
)


images_path = '/scratch/ChenhongyiYang/citypersons/image/citypersons_image/train_all'
annotations_path = '/scratch/ChenhongyiYang/citypersons/labels/train_label'


image_all = '/scratch/ChenhongyiYang/citypersons/image_all'
annotation_json = '/scratch/ChenhongyiYang/citypersons/label_all.json'



def train_cp(continue_train, is_log):
    trainer = train.Trainer(cp_train_param, cp_nets_param, cp_preprocess.preproces_for_train, image_all, annotation_json, gpus, data='cityperson')

    log_filename = time.strftime('%m_%d_%H_%M', time.localtime())

    trainer.train(continue_train, log_filename, is_log)


if __name__ == '__main__':
    if args.resume:
        if args.log:
            train_cp(continue_train=True, is_log=True)
        else:
            train_cp(continue_train=True, is_log=False)
    else:
        if args.log:
            train_cp(continue_train=False, is_log=True)
        else:
            train_cp(continue_train=False, is_log=False)









