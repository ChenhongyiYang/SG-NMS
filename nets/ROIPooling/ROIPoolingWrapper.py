# Copyright 2017 Robert Csordas. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

import tensorflow as tf
from tensorflow.python.framework import ops

try:
	roiPoolingModule = tf.load_op_library("nets/ROIPooling/roi_pooling.so")
except:
	roiPoolingModule = tf.load_op_library("./roi_pooling.so")

def positionSensitiveRoiPooling_unnorm(features, boxes, offset=[0,0], downsample=16, roiSize=3):
	with tf.name_scope("positionSensitiveRoiPooling"):
		featureCount = features.get_shape().as_list()[-1]
		
		with tf.name_scope("imgCoordinatesToHeatmapCoordinates"):
			boxes=tf.stop_gradient(boxes)
			boxes = boxes - [offset[1], offset[0], offset[1]-downsample+0.1, offset[0]-downsample+0.1]
			boxes = boxes / downsample
			boxes = tf.cast(boxes, tf.int32)

		
		with tf.name_scope("NHWC2NCHW"):
			features = tf.transpose(features, [0,3,1,2])

		res = roiPoolingModule.pos_roi_pooling(features, boxes, [roiSize,roiSize])
		
		res.set_shape([None, roiSize, roiSize, None if featureCount is None else int(featureCount/(roiSize*roiSize))])
		return res, boxes


def positionSensitiveRoiPooling_norm(features, boxes, num_channel, feature_shape, K=(3,3)):
	with tf.name_scope("positionSensitiveRoiPooling"):
		featureCount = features.get_shape().as_list()[-1]
		h, w = feature_shape

		with tf.name_scope("imgCoordinatesToHeatmapCoordinates"):
			boxes = tf.stop_gradient(boxes)
			# first x, then y
			boxes_extend = boxes + [[-0.25/w, -0.25/h, 0.25/w, 0.25/h]]
			boxes_crop = boxes_extend * [[w, h, w, h]]
			boxes_crop = tf.cast(boxes_crop, tf.int32)

		with tf.name_scope("NHWC2NCHW"):
			features = tf.transpose(features, [0, 3, 1, 2])

		res = roiPoolingModule.pos_roi_pooling(features, boxes_crop, [K[0], K[1]])

		res.set_shape(
			[None, K[0], K[1], None if featureCount is None else int(featureCount / (K[0] * K[1]))])
		res = tf.reshape(res, [-1, K[0]*K[1], num_channel])
		res = tf.transpose(res, [0, 2, 1])
		return res


@ops.RegisterGradient("PosRoiPooling")
def _pos_roi_pooling_grad(op, grad):
	g_features = roiPoolingModule.pos_roi_pooling_grad(grad, tf.shape(op.inputs[0]), op.inputs[1], op.inputs[2])
	return g_features, None, None
	