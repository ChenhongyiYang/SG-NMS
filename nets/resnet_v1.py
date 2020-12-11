import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_v2
from nets import vgg
import tensorflow.contrib.slim as slim


def resnet_101_v1(inputs,
               scope='resnet_v1_101',
               is_training=True,
               output_stride=32):
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net = resnet_v1.resnet_v1_101(inputs,
                                  num_classes=None,
                                  is_training=is_training,
                                  global_pool=False,
                                  output_stride=output_stride,
                                  reuse=None,
                                  scope=scope
                                  )
    return net

def resnet_50_v1(inputs,
              scope='resnet_v1_50',
              is_training=True,
              output_stride=32):
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net, out_dict = resnet_v1.resnet_v1_50(inputs,
                                      num_classes=None,
                                      is_training=is_training,
                                      global_pool=False,
                                      output_stride=output_stride,
                                      reuse=None,
                                      scope=scope
                                      )
        return net, out_dict


def vgg_16(inputs,
           scope='vgg_16',
           is_training=True):
    with slim.arg_scope(vgg.vgg_arg_scope()):
        nets, out_dict = vgg.vgg_16(inputs,
                                   None,
                                   is_training=is_training,
                                   spatial_squeeze=False,
                                   scope=scope,
                                   fc_conv_padding='SAME')
        return nets, out_dict



def resnet_101_v2(inputs,
               scope='resnet_v2_101',
               is_training=True,
               output_stride=32):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net = resnet_v2.resnet_v2_101(inputs,
                                      num_classes=None,
                                      is_training=is_training,
                                      global_pool=False,
                                      output_stride=output_stride,
                                      reuse=None,
                                      scope=scope
                                      )
    return net

def resnet_50_v2(inputs,
              scope='resnet_v2_50',
              is_training=True,
              output_stride=32):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net, out_dict = resnet_v2.resnet_v2_50(inputs,
                                      num_classes=None,
                                      is_training=is_training,
                                      global_pool=False,
                                      output_stride=output_stride,
                                      reuse=None,
                                      scope=scope
                                      )
        return net, out_dict

def FPN_resnet_50_v1(inputs, prefix, is_training=False):
    _, net_dict = resnet_50_v1(inputs, is_training=is_training)
    head_names = [prefix+'resnet_v1_50/block4/unit_3/bottleneck_v1',
                prefix+'resnet_v1_50/block3/unit_5/bottleneck_v1',
                prefix+'resnet_v1_50/block2/unit_3/bottleneck_v1']
    net_heads = [net_dict[name] for name in head_names]
    depths = [2048, 1024, 512]
    nets_out = []
    for i, head in enumerate(net_heads):
        if i == 0:
            nets_out.append(head)
            continue

        last_out = nets_out[i-1]
        net = slim.conv2d(last_out, depths[i], [1,1])
        net = tf.image.resize_bilinear(net, tf.shape(head)[1:3])
        net = net + head
        net = slim.conv2d(net, depths[i], [3,3])
        nets_out.append(net)
    return nets_out


def multi_resnet_50_v1(inputs, prefix, is_training=False):
    _, net_dict = resnet_50_v1(inputs, is_training=is_training)
    head_names = [prefix + 'resnet_v1_50/block4/unit_3/bottleneck_v1',
                  prefix + 'resnet_v1_50/block3/unit_5/bottleneck_v1',
                  prefix + 'resnet_v1_50/block2/unit_3/bottleneck_v1']
    net_heads = [net_dict[name] for name in head_names]
    return net_heads


def FPN_resnet_101_v1(inputs, prefix, is_training=False):
    _, net_dict = resnet_101_v1(inputs, is_training=is_training)
    head_names = [prefix+'resnet_v1_101/block4/unit_3/bottleneck_v1',
                prefix+'resnet_v1_101/block3/unit_22/bottleneck_v1',
                prefix+'resnet_v1_101/block2/unit_3/bottleneck_v1']
    net_heads = [net_dict[name] for name in head_names]
    depths = [2048, 1024, 512]
    nets_out = []
    for i, head in enumerate(net_heads):
        if i == 0:
            nets_out.append(head)
            continue

        last_out = nets_out[i-1]
        net = slim.conv2d(last_out, depths[i], [1,1])
        net = tf.image.resize_bilinear(net, tf.shape(head)[1:3])
        net = net + head
        net = slim.conv2d(net, depths[i], [3,3])
        nets_out.append(net)
    return nets_out

def FPN_resnet_50_v2(inputs, prefix, is_training=False):
    _, net_dict = resnet_50_v2(inputs, is_training=is_training)
    head_names = [prefix+'resnet_v2_50/block4/unit_3/bottleneck_v2',
                prefix+'resnet_v2_50/block3/unit_5/bottleneck_v2',
                prefix+'resnet_v2_50/block2/unit_3/bottleneck_v2']
    net_heads = [net_dict[name] for name in head_names]
    depths = [2048, 1024, 512]
    nets_out = []
    for i, head in enumerate(net_heads):
        if i == 0:
            nets_out.append(head)
            continue

        last_out = nets_out[i-1]
        net = slim.conv2d(last_out, depths[i], [1,1])
        net = tf.image.resize_bilinear(net, tf.shape(head)[1:3])
        net = net + head
        net = slim.conv2d(net, depths[i], [3,3])
        nets_out.append(net)
    return nets_out

def FPN_resnet_101_v2(inputs, prefix, is_training=False):
    _, net_dict = resnet_101_v2(inputs, is_training=is_training)
    head_names = [prefix+'resnet_v2_101/block4/unit_3/bottleneck_v2',
                prefix+'resnet_v2_101/block3/unit_22/bottleneck_v2',
                prefix+'resnet_v2_101/block2/unit_3/bottleneck_v2']
    net_heads = [net_dict[name] for name in head_names]
    depths = [2048, 1024, 512]
    nets_out = []
    for i, head in enumerate(net_heads):
        if i == 0:
            nets_out.append(head)
            continue

        last_out = nets_out[i-1]
        net = slim.conv2d(last_out, depths[i], [1,1])
        net = tf.image.resize_bilinear(net, tf.shape(head)[1:3])
        net = net + head
        net = slim.conv2d(net, depths[i], [3,3])
        nets_out.append(net)
    return nets_out







if __name__ == '__main__':
    img = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
    nets, out_dict = resnet_50_v1(img)
    for net in out_dict:
        print(net, out_dict[net])





