import tensorflow as tf
from collections import namedtuple
import numpy as np
from nets import net_class
import time
import os
import tensorflow.contrib.slim as slim
from kitti import kitti_dataset
from citypersons import cp_dataset

class moving_avg(object):
    def __init__(self,window_size):
        self.window_size = window_size
        self.values = []

    def update(self, var):
        if len(self.values) < self.window_size:
            self.values.append(var)
        else:
            self.values.pop(0)
            self.values.append(var)

    def get_avg(self):
        return float(np.mean(np.array(self.values)))

train_params = namedtuple('train_parameters', ['net',
                                            'img_shape',
                                            'net_shape',
                                            'max_box_num',
                                            'emergency_save_file',
                                            'train_log_dir',
                                            'train_model_dir',
                                            'backbone_model',
                                            'fine_tune_model',
                                            'learning_rate',
                                            'epoch_num',
                                            'log_step',
                                            'decay_rate',
                                            'decay_steps',
                                            'save_steps',
                                            'batch_size',
                                        ])

class Trainer(object):
    def __init__(self, train_param:train_params, net_param:net_class.nets_params, preprocess_fn, image_dir, annotation, gpus, data):
        self.train_param = train_param
        self.net_param = net_param
        self.preprocess_fn = preprocess_fn
        self.nets = net_class.Nets(self.net_param)


        #create tensorflow datset
        if data == 'kitti':
            self.dataset, self.batch_step = kitti_dataset.create_dataset_with_batch(image_dir, annotation, self.train_param.batch_size)
        elif data == 'cityperson':
            self.dataset, self.batch_step = cp_dataset.create_dataset_with_batch_json(image_dir, annotation, self.train_param.batch_size)
        else:
            raise ValueError('dataset error!')
        self.gpu_num = len(gpus)
        self.set_dir()

        if self.net_param.net == 'resnet_50_v1':
            self.back_net = 'resnet_v1_50'
        elif self.net_param.net == 'resnet_101_v1':
            self.back_net = 'resnet_v1_101'
        elif self.net_param.net == 'resnet_50_v2':
            self.back_net = 'resnet_v2_50'
        elif self.net_param.net == 'resnet_101_v2':
            self.back_net = 'resnet_v2_101'
        else:
            raise ValueError('Invalid backbone!')

    def set_dir(self):
        train_log_dir = self.train_param.train_log_dir
        if not os.path.isdir(train_log_dir):
            os.mkdir(train_log_dir)

        train_model_dir = self.train_param.train_model_dir
        if not os.path.isdir(train_model_dir):
            os.mkdir(train_model_dir)




    def record(self, log_file_name, during, epoch, global_step, print_names, mavgs, is_log):
        if is_log:
            f = open(os.path.join(self.train_param.train_log_dir, log_file_name + '.txt'), 'a')
            f.write('Time: %d\n' % during)
            f.write('Epoch num: %d\n' % epoch)
            f.write('Step num: %d\n' % global_step)
            for i, name in enumerate(print_names):
                f.write('%s: %5f\n' % (name, mavgs[i].get_avg()))
            f.write('\n\n')
            f.close()

        # print
        print('Time: %d' % during)
        print('Epoch num: %d' % epoch)
        print('Step num: %d' % global_step)
        for i, name in enumerate(print_names):
            print('%s: %5f' % (name, mavgs[i].get_avg()))
        print('\n')

    def average_gradients(self, tower_grads):
        """Calculate average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been
           averaged across all towers.

        reference: https://github.com/normanheckscher/mnist-multi-gpu
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def train_multiple_gpu(self, continue_train):

        iterator = self.dataset.make_initializable_iterator()
        with tf.device('/cpu:0'):

            images, labels, bboxes = iterator.get_next()
            images_sp = tf.split(images, self.gpu_num)
            labels_sp = tf.split(labels, self.gpu_num)
            bboxes_sp = tf.split(bboxes, self.gpu_num)


            tensor_mean_dict = {}

            global_step = tf.Variable(0, trainable=False, name='global_step')
            lr = tf.Variable(self.train_param.learning_rate, trainable=False, name='learning_rate', dtype=tf.float32)
            lr_decayed = tf.train.exponential_decay(lr, global_step,
                                                    decay_steps=self.train_param.decay_epoch * self.batch_step,
                                                    decay_rate=self.train_param.decay_rate,
                                                    staircase=True)

            opt = tf.train.AdamOptimizer(lr_decayed, use_locking=True)  # .minimize(total_loss, colocate_gradients_with_ops=True)
            #train_op = optimizer.minimize(total_loss, global_step, colocate_gradients_with_ops=True)

            tower_grads = []
            with tf.variable_scope(tf.get_variable_scope()):
                for gpu_id in range(self.gpu_num):
                    with tf.device('/gpu:%d'%gpu_id):
                        with tf.name_scope('Tower_%d' % gpu_id) as scope:
                            tensor_dict, print_names = self.nets.train_cascade(images_sp[gpu_id], labels_sp[gpu_id], bboxes_sp[gpu_id],
                                                                                         self.preprocess_fn,
                                                                                         self.train_param.batch_size//self.gpu_num)
                            _loss = tensor_dict['Total loss']
                            tf.get_variable_scope().reuse_variables()
                            grads = opt.compute_gradients(_loss, gate_gradients=0)
                            tower_grads.append(grads)

                            for name in print_names:
                                if name not in tensor_mean_dict:
                                    tensor_mean_dict[name] = []
                                tensor_mean_dict[name].append(tf.expand_dims(tensor_dict[name], axis=0))

            for name in print_names:
                tensor_mean_dict[name] = tf.reduce_mean(tf.concat(tensor_mean_dict[name], axis=0))

            grads = self.average_gradients(tower_grads)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies([tf.group(*update_ops)]):
                train_op = opt.apply_gradients(grads, global_step=global_step)


            #all_tensor = [n.name for n in tf.get_default_graph().as_graph_def().node]
            #for tensor in all_tensor:
            #    print(tensor)


            saver = tf.train.Saver(sharded=True)
            if not continue_train:
                def tensor_in_resnet(var, net):
                    return var.op.name.replace(net + '/'+ self.back_net, self.back_net)

                vars_adam = slim.get_variables_by_suffix("Adam") + slim.get_variables_by_suffix("Adam_1")
                vars_to_init1 = slim.get_variables_to_restore(include=[self.train_param.net + '/'+ self.back_net])
                vars_to_init1 = list(set(vars_to_init1) - set(vars_adam))
                vars_to_init1 = {tensor_in_resnet(var, self.train_param.net): var for var in vars_to_init1}

                vars_to_init2 = slim.get_variables_to_restore(exclude=[self.train_param.net + '/'+ self.back_net])
                vars_to_init2 = list(set(vars_to_init2) | set(vars_adam))
                init_op1, init_feed_dict1 = slim.assign_from_checkpoint(self.train_param.backbone_model, vars_to_init1)
                init_op2 = tf.initializers.variables(vars_to_init2)
            else:
                vars_to_init1 = slim.get_variables_to_restore(exclude=['learning_rate, global_step'])
                vars_to_init2 = slim.get_variables_to_restore(include=['learning_rate, global_step'])

                init_op1, init_feed_dict1 = slim.assign_from_checkpoint(self.train_param.fine_tune_model, vars_to_init1)
                init_op2 = tf.initializers.variables(vars_to_init2)
            return iterator, init_op1, init_feed_dict1, init_op2, saver, train_op, tensor_mean_dict, print_names

    def train_single_gpu(self, continue_train):
        iterator = self.dataset.make_initializable_iterator()
        images, labels, bboxes = iterator.get_next()
        lr = tf.placeholder(tf.float32)

        tensor_dict, print_names = self.nets.get_train_ops(images, labels,
                                                         bboxes,
                                                         self.preprocess_fn,
                                                         self.train_param.batch_size)
        total_loss = tensor_dict['Total loss']


        opt = tf.train.AdamOptimizer(lr)
        train_op = opt.minimize(total_loss)

        saver = tf.train.Saver()
        if not continue_train:
            def tensor_in_resnet(var, net):
                return var.op.name.replace(net + '/'+ self.back_net,  self.back_net)

            vars_adam = slim.get_variables_by_suffix("Adam") + slim.get_variables_by_suffix("Adam_1")
            #vars_adam = slim.get_variables_by_suffix("Momentum") + slim.get_variables_by_suffix("Momentum_1")
            vars_to_init1 = slim.get_variables_to_restore(include=[self.train_param.net + '/'+ self.back_net])
            vars_to_init1 = list(set(vars_to_init1) - set(vars_adam))
            vars_to_init1 = {tensor_in_resnet(var, self.train_param.net): var for var in vars_to_init1}

            vars_to_init2 = slim.get_variables_to_restore(exclude=[self.train_param.net + '/'+ self.back_net])
            vars_to_init2 = list(set(vars_to_init2) | set(vars_adam))
            init_op1, init_feed_dict1 = slim.assign_from_checkpoint(self.train_param.backbone_model, vars_to_init1)
            init_op2 = tf.initializers.variables(vars_to_init2)
        else:
            vars_to_init1 = slim.get_variables_to_restore(exclude=['learning_rate, global_step'])
            vars_to_init2 = slim.get_variables_to_restore(include=['learning_rate, global_step'])

            init_op1, init_feed_dict1 = slim.assign_from_checkpoint(self.train_param.fine_tune_model, vars_to_init1)
            init_op2 = tf.initializers.variables(vars_to_init2)
        return iterator, init_op1, init_feed_dict1, init_op2, saver, train_op, lr, tensor_dict, print_names

    def test(self):
        self.train_multiple_gpu(False)

    def learning_rate_schedule(self, step):
        warm_up_window = 10
        decay_window = 10000

        if step < warm_up_window:
            r = (0.9 / warm_up_window) * step + 0.1
        else:
            for i, ds in enumerate(self.train_param.decay_steps):
                if step < ds:
                    break
            if i == 0:
                r = 1.
            else:
                delta = step - self.train_param.decay_steps[i - 1]
                if delta < decay_window:
                    _r = 1 - ((1 - self.train_param.decay_rate) / decay_window) * delta
                    r = self.train_param.decay_rate**(i-1.) * _r
                else:
                    r = self.train_param.decay_rate**i
        return r


    def train(self, continue_train, log_file_name, is_log):
        #iterator, init_op1, init_feed_dict1, init_op2, saver, train_op, tensor_mean_dict, print_names = self.train_multiple_gpu(continue_train)
        iterator, init_op1, init_feed_dict1, init_op2, saver, train_op, lr, tensor_mean_dict, print_names = self.train_single_gpu(continue_train)
        tensors_to_run = [train_op]
        for name in print_names:
            tensors_to_run.append(tensor_mean_dict[name])

        window_size = self.train_param.log_step
        supervised_num = len(print_names)
        mavgs = [moving_avg(window_size) for _ in range(supervised_num)]

        global_step = 0
        start_time = time.time()
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            if not continue_train:
                sess.run(init_op1, init_feed_dict1)
                sess.run(init_op2)
                sess.run(iterator.initializer)
            else:
                sess.run(iterator.initializer)
                saver.restore(sess, self.train_param.fine_tune_model)

            global_step += 1

            for epoch in range(self.train_param.epoch_num):
                for step in range(self.batch_step):
                    r = self.learning_rate_schedule(global_step)
                    global_step += 1
                    results_list = sess.run(tensors_to_run, feed_dict={lr: r * self.train_param.learning_rate})

                    for i in range(1, len(results_list)):
                        mavgs[i - 1].update(results_list[i])

                    if global_step % self.train_param.log_step == 0:
                        self.record(log_file_name, time.time() - start_time, epoch, global_step, print_names, mavgs, is_log)

                    if global_step in self.train_param.save_steps:
                        saver.save(sess, os.path.join(self.train_param.train_model_dir,'model_' + log_file_name.split('.')[0] + '.ckpt'), global_step=global_step)
                        print('model_' + log_file_name.split('.')[0] + '.ckpt' + ' is saved!!')

                try:
                    ef = open(self.train_param.emergency_save_file, 'r')
                    e_epoch = int(ef.readline().strip())
                    ef.close()
                except:
                    e_epoch = 999

                saver.save(sess, os.path.join(self.train_param.train_model_dir,'latestModel_' + log_file_name.split('.')[0] + '.ckpt'))



















