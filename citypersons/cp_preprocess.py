import tensorflow as tf
from preprocessing import tf_image
from preprocessing import preprocessing_rfcn as pr



def preproces_for_train(image, labels, bboxes, input_shape, crop_shape, out_shape):
    image, labels, bboxes = local_preprocess_for_train(image, labels, bboxes, input_shape, crop_shape, out_shape)
    return image, labels, bboxes

def preprocess_for_eval(image,out_shape):
    img = tf.image.convert_image_dtype(image, tf.float32)
    img = tf_image.resize_image(img, out_shape,
                                method=tf.image.ResizeMethod.BILINEAR,
                                align_corners=False)
    img = resnet_norm(img)
    return img

def resnet_norm(img, mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)):
    mean_tensor = tf.constant(mean, tf.float32)
    std_tensor = tf.constant(std, tf.float32)
    ret_img = tf.div(img - mean_tensor, std_tensor)
    return ret_img

def resnet_denorm(img, mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)):
    mean_tensor = tf.constant(mean, tf.float32)
    std_tensor = tf.constant(std, tf.float32)
    ret_img = img * std_tensor + mean_tensor
    return ret_img


def vgg_norm(img):
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    _NUM_CHANNELS = 3
    means = (_R_MEAN, _G_MEAN, _B_MEAN)
    channels = tf.split(axis=2, num_or_size_splits=_NUM_CHANNELS, value=img)
    for i in range(_NUM_CHANNELS):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)


def local_preprocess_for_train(image, labels, bboxes, input_shape, crop_shape,
                               out_shape, data_format='NHWC',
                               scope='rfcn_preprocessing_train'):

    fast_mode = False
    with tf.name_scope(scope, 'rfcn_preprocessing_train', [image, labels, bboxes]):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')
        # Convert to float scaled [0, 1].
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        dst_image, dst_labels, dst_bboxes = pr.random_crop(image, labels, bboxes, img_shape=input_shape, crop_shape=crop_shape, keep_ratio=0.7, random_scale=(0.7, 1.3))

        dst_image = tf.image.resize(dst_image, (out_shape[0], out_shape[1]))

        dst_image, dst_bboxes = tf_image.random_flip_left_right(dst_image, dst_bboxes)

        #dst_image = pr.apply_with_random_selector(dst_image,lambda x, ordering: pr.distort_color(x, ordering, fast_mode),num_cases=4)

        dst_image = pr.resnet_norm(dst_image)

        if data_format == 'NCHW':
            dst_image = tf.transpose(image, perm=(2, 0, 1))
        return dst_image , dst_labels, dst_bboxes











































