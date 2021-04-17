from __future__ import print_function
import tensorflow as tf
import numpy as np
import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
from cnn_net import *
import BatchDatsetReader as dataset
from six.moves import xrange
import time
import visualize as vis
from guided_filter import guided_filter
import os
from deeplab import *
tf.reset_default_graph()
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "4", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "data3/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "test", "Mode train/ test/ visualize")
MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'
MAX_ITERATION = int(1133)
NUM_OF_CLASSESS = 17
IMAGE_SIZE = 224
use_weight_regularizer = True
test_epoch = 132
decay_rate = 0.95 # 衰减率
decay_steps = 500# 衰减次数
def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net
def inference_yuanshi(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        pool5 = utils.max_pool_2x2(image_net["conv5_3"])
        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)
        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)
        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")
        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")
        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], 224, 224, NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)
        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")
    return tf.expand_dims(annotation_pred, dim=3), conv_t3
def inference_yunshi_jia1(image,  keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)
    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = np.squeeze(model_data['layers'])
    processed_image = utils.process_image(image, mean_pixel)
    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        pool5 = tf.layers.max_pooling2d(image_net["conv5_3"], 2, 2)
        relu6 = tf.layers.conv2d(pool5, 4096, (7, 7), activation="relu",padding="same", name="relu6")
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)
        relu7 = tf.layers.conv2d(relu_dropout6, 4096, (1, 1), activation="relu", padding="same", name="relu7")
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)
        conv8 = tf.layers.conv2d(relu_dropout7, NUM_OF_CLASSESS, (1, 1), padding="same", name="conv8")
        print("conv8.shape", conv8.shape)
        # 卷积第一次
        deconv_shape1 = image_net["pool4"].get_shape()
        conv_t1 = tf.layers.conv2d_transpose(conv8, deconv_shape1[3].value, (4, 4), strides=2, padding="same")
        print("conv_t1.shape:", conv_t1.shape)
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")
        print("fuse_1_conv.shape", fuse_1.shape)
        deconv_shape2 = image_net["pool3"].get_shape()
        conv_t2 = tf.layers.conv2d_transpose(fuse_1, deconv_shape2[3].value, (4, 4), strides=2, padding="same")
        fuse_2 = tf.add(conv_t2, image_net["pool3"],  name="fuse_2")
        print("fuse_2_conv.shape", fuse_2.shape)
        deconv_shape3 = image_net["pool2"].get_shape()
        conv_t3 = tf.layers.conv2d_transpose(fuse_2, deconv_shape3[3].value, (4, 4), strides=2, padding="same")
        fuse_3 = tf.add(conv_t3, image_net["pool2"], name="fuse_3")
        conv_t2_4 = tf.layers.conv2d_transpose(fuse_3, NUM_OF_CLASSESS, (8, 8), strides=4, padding="same")
        annotation_pred = tf.argmax(conv_t2_4, dimension=3, name="prediction")
    return tf.expand_dims(annotation_pred, dim=3), conv_t2_4
def inference_yunshi_jia1_conv3x3(image,  keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)
    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = np.squeeze(model_data['layers'])
    processed_image = utils.process_image(image, mean_pixel)
    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        pool5 = tf.layers.max_pooling2d(image_net["conv5_3"], 2, 2)
        relu6 = tf.layers.conv2d(pool5, 4096, (7, 7), activation="relu",padding="same", name="relu6")
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)
        relu7 = tf.layers.conv2d(relu_dropout6, 4096, (1, 1), activation="relu", padding="same", name="relu7")
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)
        conv8 = tf.layers.conv2d(relu_dropout7, NUM_OF_CLASSESS, (1, 1), padding="same", name="conv8")
        print("conv8.shape", conv8.shape)
        # 卷积第一次
        deconv_shape1 = image_net["pool4"].get_shape()
        conv_t1 = tf.layers.conv2d_transpose(conv8, deconv_shape1[3].value, (4, 4), strides=2, padding="same")
        print("conv_t1.shape:", conv_t1.shape)
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")
        fuse_1 = tf.layers.conv2d(fuse_1, deconv_shape1[3].value, (3, 3), activation="relu", padding="same")
        fuse_1 = tf.layers.conv2d(fuse_1, deconv_shape1[3].value, (3, 3), activation="relu", padding="same")
        print("fuse_1_conv.shape", fuse_1.shape)
        deconv_shape2 = image_net["pool3"].get_shape()
        conv_t2 = tf.layers.conv2d_transpose(fuse_1, deconv_shape2[3].value, (4, 4), strides=2, padding="same")
        fuse_2 = tf.add(conv_t2, image_net["pool3"],  name="fuse_2")
        fuse_2 = tf.layers.conv2d(fuse_2, deconv_shape2[3].value, (3, 3), activation="relu", padding="same")
        fuse_2 = tf.layers.conv2d(fuse_2, deconv_shape2[3].value, (3, 3), activation="relu", padding="same")
        print("fuse_2_conv.shape", fuse_2.shape)
        deconv_shape3 = image_net["pool2"].get_shape()
        conv_t3 = tf.layers.conv2d_transpose(fuse_2, deconv_shape3[3].value, (4, 4), strides=2, padding="same")
        fuse_3 = tf.add(conv_t3, image_net["pool2"], name="fuse_3")
        fuse_3 = tf.layers.conv2d(fuse_3, deconv_shape3[3].value, (3, 3), activation="relu", padding="same")
        fuse_3 = tf.layers.conv2d(fuse_3, deconv_shape3[3].value, (3, 3), activation="relu", padding="same")
        conv_t2_4 = tf.layers.conv2d_transpose(fuse_3, NUM_OF_CLASSESS, (8, 8), strides=4, padding="same")
        annotation_pred = tf.argmax(conv_t2_4, dimension=3, name="prediction")
    return tf.expand_dims(annotation_pred, dim=3), conv_t2_4
def inference_yunshi_jia1_fpn(image,  keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)
    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = np.squeeze(model_data['layers'])
    processed_image = utils.process_image(image, mean_pixel)
    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        pool5 = tf.layers.max_pooling2d(image_net["conv5_3"], 2, 2)
        relu6 = tf.layers.conv2d(pool5, 4096, (7, 7), activation="relu",padding="same", name="relu6")
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)
        relu7 = tf.layers.conv2d(relu_dropout6, 4096, (1, 1), activation="relu", padding="same", name="relu7")
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)
        conv8 = tf.layers.conv2d(relu_dropout7, NUM_OF_CLASSESS, (1, 1), padding="same", name="conv8")
        print("conv8.shape", conv8.shape)
        # 卷积第一次
        deconv_shape1 = image_net["pool4"].get_shape()
        conv_t1 = tf.layers.conv2d_transpose(conv8, deconv_shape1[3].value, (4, 4), strides=2, padding="same")
        print("conv_t1.shape:", conv_t1.shape)
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")
        # fuse_1 = tf.layers.conv2d(fuse_1, deconv_shape1[3].value, (3, 3), activation="relu", padding="same")
        # fuse_1 = tf.layers.conv2d(fuse_1, deconv_shape1[3].value, (3, 3), activation="relu", padding="same")
        print("fuse_1.shape", fuse_1.shape)
        #卷积第二次
        deconv_shape2 = image_net["pool3"].get_shape()
        conv_t2 = tf.layers.conv2d_transpose(fuse_1, deconv_shape2[3].value, (4, 4), strides=2, padding="same")
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")
        # fuse_2 = tf.layers.conv2d(fuse_2, deconv_shape2[3].value, (3, 3), activation="relu", padding="same")
        # fuse_2 = tf.layers.conv2d(fuse_2, deconv_shape2[3].value, (3, 3), activation="relu", padding="same")
    #反卷积 8s
        #卷积第三次
        deconv_shape3 = image_net["pool2"].get_shape()
        conv_t3 = tf.layers.conv2d_transpose(fuse_2, deconv_shape3[3].value, (4, 4), strides=2, padding="same")
        fuse_3 = tf.add(conv_t3, image_net["pool2"], name="fuse_3")
        # fuse_3 = tf.layers.conv2d(fuse_3, deconv_shape1[3].value, (3, 3), activation="relu", padding="same")
        # fuse_3 = tf.layers.conv2d(fuse_3, deconv_shape1[3].value, (3, 3), activation="relu", padding="same")
        conv_t2_4 = tf.layers.conv2d_transpose(fuse_3, NUM_OF_CLASSESS, (8, 8), strides=4, padding="same")
        print("conv_t2_4.shape:", conv_t2_4.shape)
        feature_5 = image_net['conv5_3']
        print("feature_5.shape", feature_5.shape)
        feature_5_5 = tf.layers.conv2d_transpose(feature_5, NUM_OF_CLASSESS, (32, 32), strides=16, padding="same")
        feature_4 = image_net['relu4_4']
        print("feature_4.shape", feature_4.shape)
        feature_shape_4 = image_net['relu4_4'].get_shape()
        feature_4_1 = tf.layers.conv2d_transpose(feature_5, feature_shape_4[3].value, (4, 4), strides=2, padding="same")
        feature_4 = tf.add(feature_4_1, feature_4)
        feature_4_4 = tf.layers.conv2d_transpose(feature_4, NUM_OF_CLASSESS, (16, 16), strides=8, padding="same")
        print("feature_4_logits.shape", feature_4_4.shape)
        feature_3 = image_net['relu3_4']
        print("feature_3.shape", feature_3.shape)
        feature_shape_3 = image_net['relu3_4'].get_shape()
        feature_3_1 = tf.layers.conv2d_transpose(feature_4, feature_shape_3[3].value, (4, 4), strides=2, padding="same")
        feature_3 = tf.add(feature_3_1, feature_3)
        feature_3_3 = tf.layers.conv2d_transpose(feature_3, NUM_OF_CLASSESS, (8, 8), strides=4, padding="same")
        print("feature_3_logits.shape", feature_3_3.shape)
        feature_2 = image_net['relu2_2']
        feature_shape_2 = image_net['relu2_2'].get_shape()
        feature_2_1 = tf.layers.conv2d_transpose(feature_3, feature_shape_2[3].value, (4, 4), strides=2, padding="same")
        feature_2 = tf.add(feature_2, feature_2_1)
        print("feature_2.shape", feature_2.shape)
        feature_2_2 = tf.layers.conv2d_transpose(feature_2, NUM_OF_CLASSESS, (4, 4), strides=2, padding="same")
        print("feature_2_logits.shape", feature_2_2.shape)
        feature_1 = image_net['relu1_2']
        print("feature_1.shape", feature_1.shape)
        feature_shape_1 = image_net['relu1_2'].get_shape()
        feature_1_1 = tf.layers.conv2d_transpose(feature_2, feature_shape_1[3].value, (4, 4), strides=2, padding="same")
        feature_1 = tf.add(feature_1, feature_1_1)
        feature_1_1_1 = tf.layers.conv2d(feature_1, NUM_OF_CLASSESS, (1, 1), padding="same")
        print("feature_1_logits.shape", feature_1_1.shape)
        merge_all = tf.concat(values=[feature_1_1_1, feature_2_2, feature_3_3, feature_4_4, feature_5_5, conv_t2_4], axis=3, name="merge_all")
        merge_all_1 = tf.layers.conv2d(merge_all, NUM_OF_CLASSESS, (1, 1), activation="relu", padding="same")
        annotation_pred_4 = tf.argmax(merge_all_1, dimension=3, name="prediction", output_type=tf.int32)
    return tf.expand_dims(annotation_pred_4, dim=3), merge_all_1
def inference_yunshi_jia1_fpn_conv3x3(image,  keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)
    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = np.squeeze(model_data['layers'])
    processed_image = utils.process_image(image, mean_pixel)
    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        pool5 = tf.layers.max_pooling2d(image_net["conv5_3"], 2, 2)
        relu6 = tf.layers.conv2d(pool5, 4096, (7, 7), activation="relu",padding="same", name="relu6")
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)
        relu7 = tf.layers.conv2d(relu_dropout6, 4096, (1, 1), activation="relu", padding="same", name="relu7")
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)
        conv8 = tf.layers.conv2d(relu_dropout7, NUM_OF_CLASSESS, (1, 1), padding="same", name="conv8")
        print("conv8.shape", conv8.shape)
        # 卷积第一次
        deconv_shape1 = image_net["pool4"].get_shape()
        conv_t1 = tf.layers.conv2d_transpose(conv8, deconv_shape1[3].value, (4, 4), strides=2, padding="same")
        print("conv_t1.shape:", conv_t1.shape)
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")
        fuse_1 = tf.layers.conv2d(fuse_1, deconv_shape1[3].value, (3, 3), activation="relu", padding="same")
        fuse_1 = tf.layers.conv2d(fuse_1, deconv_shape1[3].value, (3, 3), activation="relu", padding="same")
        print("fuse_1.shape", fuse_1.shape)
        #卷积第二次
        deconv_shape2 = image_net["pool3"].get_shape()
        conv_t2 = tf.layers.conv2d_transpose(fuse_1, deconv_shape2[3].value, (4, 4), strides=2, padding="same")
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")
        fuse_2 = tf.layers.conv2d(fuse_2, deconv_shape2[3].value, (3, 3), activation="relu", padding="same")
        fuse_2 = tf.layers.conv2d(fuse_2, deconv_shape2[3].value, (3, 3), activation="relu", padding="same")
    #反卷积 8s
        #卷积第三次
        deconv_shape3 = image_net["pool2"].get_shape()
        conv_t3 = tf.layers.conv2d_transpose(fuse_2, deconv_shape3[3].value, (4, 4), strides=2, padding="same")
        fuse_3 = tf.add(conv_t3, image_net["pool2"], name="fuse_3")
        fuse_3 = tf.layers.conv2d(fuse_3, deconv_shape3[3].value, (3, 3), activation="relu", padding="same")
        fuse_3 = tf.layers.conv2d(fuse_3, deconv_shape3[3].value, (3, 3), activation="relu", padding="same")

        conv_t2_4 = tf.layers.conv2d_transpose(fuse_3, NUM_OF_CLASSESS, (8, 8), strides=4, padding="same")
        print("conv_t2_4.shape:", conv_t2_4.shape)
        feature_5 = image_net['conv5_3']
        print("feature_5.shape", feature_5.shape)
        feature_5_5 = tf.layers.conv2d_transpose(feature_5, NUM_OF_CLASSESS, (32, 32), strides=16, padding="same")
        feature_4 = image_net['relu4_4']
        print("feature_4.shape", feature_4.shape)
        feature_shape_4 = image_net['relu4_4'].get_shape()
        feature_4_1 = tf.layers.conv2d_transpose(feature_5, feature_shape_4[3].value, (4, 4), strides=2, padding="same")
        feature_4 = tf.add(feature_4_1, feature_4)
        feature_4_4 = tf.layers.conv2d_transpose(feature_4, NUM_OF_CLASSESS, (16, 16), strides=8, padding="same")
        print("feature_4_logits.shape", feature_4_4.shape)
        feature_3 = image_net['relu3_4']
        print("feature_3.shape", feature_3.shape)
        feature_shape_3 = image_net['relu3_4'].get_shape()
        feature_3_1 = tf.layers.conv2d_transpose(feature_4, feature_shape_3[3].value, (4, 4), strides=2, padding="same")
        feature_3 = tf.add(feature_3_1, feature_3)
        feature_3_3 = tf.layers.conv2d_transpose(feature_3, NUM_OF_CLASSESS, (8, 8), strides=4, padding="same")
        print("feature_3_logits.shape", feature_3_3.shape)
        feature_2 = image_net['relu2_2']
        feature_shape_2 = image_net['relu2_2'].get_shape()
        feature_2_1 = tf.layers.conv2d_transpose(feature_3, feature_shape_2[3].value, (4, 4), strides=2, padding="same")
        feature_2 = tf.add(feature_2, feature_2_1)
        print("feature_2.shape", feature_2.shape)
        feature_2_2 = tf.layers.conv2d_transpose(feature_2, NUM_OF_CLASSESS, (4, 4), strides=2, padding="same")
        print("feature_2_logits.shape", feature_2_2.shape)
        feature_1 = image_net['relu1_2']
        print("feature_1.shape", feature_1.shape)
        feature_shape_1 = image_net['relu1_2'].get_shape()
        feature_1_1 = tf.layers.conv2d_transpose(feature_2, feature_shape_1[3].value, (4, 4), strides=2, padding="same")
        feature_1 = tf.add(feature_1, feature_1_1)
        feature_1_1_1 = tf.layers.conv2d(feature_1, NUM_OF_CLASSESS, (1, 1), padding="same")
        print("feature_1_logits.shape", feature_1_1.shape)
        merge_all = tf.concat(values=[feature_1_1_1, feature_2_2, feature_3_3, feature_4_4, feature_5_5, conv_t2_4], axis=3, name="merge_all")
        merge_all_1 = tf.layers.conv2d(merge_all, NUM_OF_CLASSESS, (1, 1), activation="relu", padding="same")
        annotation_pred_4 = tf.argmax(merge_all_1, dimension=3, name="prediction", output_type=tf.int32)
    return tf.expand_dims(annotation_pred_4, dim=3), merge_all_1
def inference_yunshi_jia1_feature_paymid(image,  keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)
    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = np.squeeze(model_data['layers'])
    processed_image = utils.process_image(image, mean_pixel)
    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        pool5 = tf.layers.max_pooling2d(image_net["conv5_3"], 2, 2)
        relu6 = tf.layers.conv2d(pool5, 4096, (7, 7), activation="relu",padding="same", name="relu6")
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)
        relu7 = tf.layers.conv2d(relu_dropout6, 4096, (1, 1), activation="relu", padding="same", name="relu7")
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)
        conv8 = tf.layers.conv2d(relu_dropout7, NUM_OF_CLASSESS, (1, 1), padding="same", name="conv8")
        print("conv8.shape", conv8.shape)
        # 卷积第一次
        deconv_shape1 = image_net["pool4"].get_shape()
        conv_t1 = tf.layers.conv2d_transpose(conv8, deconv_shape1[3].value, (4, 4), strides=2, padding="same")
        print("conv_t1.shape:", conv_t1.shape)
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")
        print("fuse_1_conv.shape", fuse_1.shape)
        #卷积第二次
        deconv_shape2 = image_net["pool3"].get_shape()
        conv_t2 = tf.layers.conv2d_transpose(fuse_1, deconv_shape2[3].value, (4, 4), strides=2, padding="same")
        fuse_2 = tf.add(conv_t2, image_net["pool3"],  name="fuse_2")
        print("fuse_2_conv.shape", fuse_2.shape)
        #卷积第三次
        deconv_shape3 = image_net["pool2"].get_shape()
        conv_t3 = tf.layers.conv2d_transpose(fuse_2, deconv_shape3[3].value, (4, 4), strides=2, padding="same")
        fuse_3 = tf.add(conv_t3, image_net["pool2"], name="fuse_3")
        conv_t2_4 = tf.layers.conv2d_transpose(fuse_3, NUM_OF_CLASSESS, (8, 8), strides=4, padding="same")
        print("conv_t2_4.shape:", conv_t2_4.shape)
        feature_1 = image_net['relu1_2']
        print("feature_1.shape", feature_1.shape)
        feature_1_1 = tf.layers.conv2d(feature_1, NUM_OF_CLASSESS, (1, 1), activation="relu", padding="same")
        print("feature_1_logits.shape", feature_1_1.shape)
        feature_2 = image_net['relu2_2']
        feature_2_2 = tf.layers.conv2d_transpose(feature_2, NUM_OF_CLASSESS, (4, 4), strides=2, padding="same")
        print("feature_2_logits.shape", feature_2_2.shape)
        feature_3 = image_net['relu3_4']
        feature_3_3 = tf.layers.conv2d_transpose(feature_3, NUM_OF_CLASSESS, (8, 8), strides=4, padding="same")
        print("feature_3_logits.shape", feature_3_3.shape)
        feature_4 = image_net['relu4_4']
        feature_4_4 = tf.layers.conv2d_transpose(feature_4, NUM_OF_CLASSESS, (16, 16), strides=8, padding="same")
        print("feature_4_logits.shape", feature_4_4.shape)
        feature_5 = image_net['conv5_3']
        print("feature_5.shape", feature_5.shape)
        feature_5_5 = tf.layers.conv2d_transpose(feature_5, NUM_OF_CLASSESS, (32, 32), strides=16, padding="same")
        print("feature_5_logits.shape", feature_5_5.shape)
        merge_all = tf.concat(values=[feature_1_1, feature_2_2, feature_3_3, feature_4_4, feature_5_5, conv_t2_4], axis=3, name="merge_all")
        merge_all_1 = tf.layers.conv2d(merge_all, NUM_OF_CLASSESS, (1, 1), activation="relu", padding="same")
        annotation_pred_4 = tf.argmax(merge_all_1, dimension=3, name="prediction", output_type=tf.int32)
    return tf.expand_dims(annotation_pred_4, dim=3), merge_all_1
def inference_yunshi_jia1_feature_paymid_conv3x3(image,  keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)
    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = np.squeeze(model_data['layers'])
    processed_image = utils.process_image(image, mean_pixel)
    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        pool5 = tf.layers.max_pooling2d(image_net["conv5_3"], 2, 2)
        relu6 = tf.layers.conv2d(pool5, 4096, (7, 7), activation="relu",padding="same", name="relu6")
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)
        relu7 = tf.layers.conv2d(relu_dropout6, 4096, (1, 1), activation="relu", padding="same", name="relu7")
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)
        conv8 = tf.layers.conv2d(relu_dropout7, NUM_OF_CLASSESS, (1, 1), padding="same", name="conv8")
        print("conv8.shape", conv8.shape)
        # 卷积第一次
        deconv_shape1 = image_net["pool4"].get_shape()
        conv_t1 = tf.layers.conv2d_transpose(conv8, deconv_shape1[3].value, (4, 4), strides=2, padding="same")
        print("conv_t1.shape:", conv_t1.shape)
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")
        fuse_1 = tf.layers.conv2d(fuse_1, deconv_shape1[3].value, (3, 3), activation="relu", padding="same")
        fuse_1 = tf.layers.conv2d(fuse_1, deconv_shape1[3].value, (3, 3), activation="relu", padding="same")
        print("fuse_1_conv.shape", fuse_1.shape)
        #卷积第二次
        deconv_shape2 = image_net["pool3"].get_shape()
        conv_t2 = tf.layers.conv2d_transpose(fuse_1, deconv_shape2[3].value, (4, 4), strides=2, padding="same")
        fuse_2 = tf.add(conv_t2, image_net["pool3"],  name="fuse_2")
        fuse_2 = tf.layers.conv2d(fuse_2, deconv_shape2[3].value, (3, 3), activation="relu", padding="same")
        fuse_2 = tf.layers.conv2d(fuse_2, deconv_shape2[3].value, (3, 3), activation="relu", padding="same")
        print("fuse_2_conv.shape", fuse_2.shape)
        #卷积第三次
        deconv_shape3 = image_net["pool2"].get_shape()
        conv_t3 = tf.layers.conv2d_transpose(fuse_2, deconv_shape3[3].value, (4, 4), strides=2, padding="same")
        fuse_3 = tf.add(conv_t3, image_net["pool2"], name="fuse_3")
        fuse_3 = tf.layers.conv2d(fuse_3, deconv_shape3[3].value, (3, 3), activation="relu", padding="same")
        fuse_3 = tf.layers.conv2d(fuse_3, deconv_shape3[3].value, (3, 3), activation="relu", padding="same")
        conv_t2_4 = tf.layers.conv2d_transpose(fuse_3, NUM_OF_CLASSESS, (8, 8), strides=4, padding="same")
        print("conv_t2_4.shape:", conv_t2_4.shape)
        feature_1 = image_net['relu1_2']
        print("feature_1.shape", feature_1.shape)
        feature_1_1 = tf.layers.conv2d(feature_1, NUM_OF_CLASSESS, (1, 1), activation="relu", padding="same")
        print("feature_1_logits.shape", feature_1_1.shape)
        feature_2 = image_net['relu2_2']
        feature_2_2 = tf.layers.conv2d_transpose(feature_2, NUM_OF_CLASSESS, (4, 4), strides=2, padding="same")
        print("feature_2_logits.shape", feature_2_2.shape)
        feature_3 = image_net['relu3_4']
        feature_3_3 = tf.layers.conv2d_transpose(feature_3, NUM_OF_CLASSESS, (8, 8), strides=4, padding="same")
        print("feature_3_logits.shape", feature_3_3.shape)
        feature_4 = image_net['relu4_4']
        feature_4_4 = tf.layers.conv2d_transpose(feature_4, NUM_OF_CLASSESS, (16, 16), strides=8, padding="same")
        print("feature_4_logits.shape", feature_4_4.shape)
        feature_5 = image_net['conv5_3']
        print("feature_5.shape", feature_5.shape)
        feature_5_5 = tf.layers.conv2d_transpose(feature_5, NUM_OF_CLASSESS, (32, 32), strides=16, padding="same")
        print("feature_5_logits.shape", feature_5_5.shape)
        merge_all = tf.concat(values=[feature_1_1, feature_2_2, feature_3_3, feature_4_4, feature_5_5, conv_t2_4], axis=3, name="merge_all")
        merge_all_1 = tf.layers.conv2d(merge_all, NUM_OF_CLASSESS, (1, 1), activation="relu", padding="same")
        annotation_pred_4 = tf.argmax(merge_all_1, dimension=3, name="prediction", output_type=tf.int32)
    return tf.expand_dims(annotation_pred_4, dim=3), merge_all_1
def inference_yunshi_jia1_feature_paymid_conv3x3_per_guided(image,binary_image,keep_prob):
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)
    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = np.squeeze(model_data['layers'])
    processed_image = utils.process_image(image, mean_pixel)
    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        binary_image = tf.layers.conv2d(binary_image, filters=3, kernel_size=(1, 1),
                                        strides=1, padding='same',
                                        activation=None)
        pool5 = tf.layers.max_pooling2d(image_net["conv5_3"], 2, 2)
        relu6 = tf.layers.conv2d(pool5, 4096, (7, 7), activation="relu",padding="same", name="relu6")
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)
        relu7 = tf.layers.conv2d(relu_dropout6, 4096, (1, 1), activation="relu", padding="same", name="relu7")
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)
        conv8 = tf.layers.conv2d(relu_dropout7, NUM_OF_CLASSESS, (1, 1), padding="same", name="conv8")
        print("conv8.shape", conv8.shape)
        # 卷积第一次
        deconv_shape1 = image_net["pool4"].get_shape()
        conv_t1 = tf.layers.conv2d_transpose(conv8, deconv_shape1[3].value, (4, 4), strides=2, padding="same")
        print("conv_t1.shape:", conv_t1.shape)
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")
        fuse_1 = tf.layers.conv2d(fuse_1, deconv_shape1[3].value, (3, 3), activation="relu", padding="same")
        fuse_1 = tf.layers.conv2d(fuse_1, deconv_shape1[3].value, (3, 3), activation="relu", padding="same")
        print("fuse_1_conv.shape", fuse_1.shape)
        ##################################################################################################
        # image_1 = tf.image.resize_images(image, (deconv_shape1[1].value, deconv_shape1[2].value))
        # binary_image_1 = tf.image.resize_images(binary_image, (deconv_shape1[1].value, deconv_shape1[2].value))
        # binary_image_1 = tf.cast(binary_image_1, tf.float32)
        # print("binary_image_1.shape:", binary_image_1.shape)
        # merge_1 = tf.concat([image_1, binary_image_1], axis=3)
        # guided_post_1 = tf.layers.conv2d(merge_1, filters=32, kernel_size=(1, 1),
        #                                  strides=1, padding='same',
        #                                  activation="relu")
        # guided_post_1 = tf.layers.conv2d(guided_post_1, filters=deconv_shape1[3].value, kernel_size=(1, 1),
        #                                  strides=1, padding='same',
        #                                  activation=None)
        # print("guided_post_1.shape:", guided_post_1.shape)
        # fuse_1 = guided_filter(guided_post_1, fuse_1, 6, 0.01, nhwc=True)
        ########################################################################################################

        #卷积第二次
        deconv_shape2 = image_net["pool3"].get_shape()
        conv_t2 = tf.layers.conv2d_transpose(fuse_1, deconv_shape2[3].value, (4, 4), strides=2, padding="same")
        fuse_2 = tf.add(conv_t2, image_net["pool3"],  name="fuse_2")
        fuse_2 = tf.layers.conv2d(fuse_2, deconv_shape2[3].value, (3, 3), activation="relu", padding="same")
        fuse_2 = tf.layers.conv2d(fuse_2, deconv_shape2[3].value, (3, 3), activation="relu", padding="same")
        print("fuse_2_conv.shape", fuse_2.shape)
        ####################################################################################################
        # image_2 = tf.image.resize_images(image, (deconv_shape2[1].value, deconv_shape2[2].value))
        # binary_image_2 = tf.image.resize_images(binary_image, (deconv_shape2[1].value, deconv_shape2[2].value))
        # binary_image_2 = tf.cast(binary_image_2, tf.float32)
        # print("binary_image_1.shape:", binary_image_2.shape)
        # merge_2 = tf.concat([image_2, binary_image_2], axis=3)
        # guided_post_2 = tf.layers.conv2d(merge_2, filters=32, kernel_size=(1, 1),
        #                                  strides=1, padding='same',
        #                                  activation="relu")
        # guided_post_2 = tf.layers.conv2d(guided_post_2, filters=deconv_shape2[3].value, kernel_size=(1, 1),
        #                                  strides=1, padding='same',
        #                                  activation=None)
        # fuse_2 = guided_filter(guided_post_2, fuse_2, 6, 0.01, nhwc=True)
        ######################################################################################################
        #卷积第三次
        deconv_shape3 = image_net["pool2"].get_shape()
        conv_t3 = tf.layers.conv2d_transpose(fuse_2, deconv_shape3[3].value, (4, 4), strides=2, padding="same")
        fuse_3 = tf.add(conv_t3, image_net["pool2"], name="fuse_3")
        fuse_3 = tf.layers.conv2d(fuse_3, deconv_shape3[3].value, (3, 3), activation="relu", padding="same")
        fuse_3 = tf.layers.conv2d(fuse_3, deconv_shape3[3].value, (3, 3), activation="relu", padding="same")
        ##########################################################################################################
        image_3 = tf.image.resize_images(image, (deconv_shape3[1].value, deconv_shape3[2].value))
        binary_image_3 = tf.image.resize_images(binary_image, (deconv_shape3[1].value, deconv_shape3[2].value))
        binary_image_3 = tf.cast(binary_image_3, tf.float32)
        print("binary_image_3.shape:", binary_image_3.shape)
        merge_3 = tf.concat([image_3, binary_image_3], axis=3)
        guided_post_3 = tf.layers.conv2d(merge_3, filters=32, kernel_size=(1, 1),
                                         strides=1, padding='same',
                                         activation="relu")
        guided_post_3 = tf.layers.conv2d(guided_post_3, filters=deconv_shape3[3].value, kernel_size=(1, 1),
                                         strides=1, padding='same',
                                         activation=None)
        fuse_3 = guided_filter(guided_post_3, fuse_3, 6, 0.01, nhwc=True)
       #########################################################################################################
        conv_t2_4 = tf.layers.conv2d_transpose(fuse_3, NUM_OF_CLASSESS, (8, 8), strides=4, padding="same")
        print("conv_t2_4.shape:", conv_t2_4.shape)
        feature_1 = image_net['relu1_2']
        print("feature_1.shape", feature_1.shape)
        feature_1_1 = tf.layers.conv2d(feature_1, NUM_OF_CLASSESS, (1, 1), activation="relu", padding="same")
        print("feature_1_logits.shape", feature_1_1.shape)
        feature_2 = image_net['relu2_2']
        feature_2_2 = tf.layers.conv2d_transpose(feature_2, NUM_OF_CLASSESS, (4, 4), strides=2, padding="same")
        print("feature_2_logits.shape", feature_2_2.shape)
        feature_3 = image_net['relu3_4']
        feature_3_3 = tf.layers.conv2d_transpose(feature_3, NUM_OF_CLASSESS, (8, 8), strides=4, padding="same")
        print("feature_3_logits.shape", feature_3_3.shape)
        feature_4 = image_net['relu4_4']
        feature_4_4 = tf.layers.conv2d_transpose(feature_4, NUM_OF_CLASSESS, (16, 16), strides=8, padding="same")
        print("feature_4_logits.shape", feature_4_4.shape)
        feature_5 = image_net['conv5_3']
        print("feature_5.shape", feature_5.shape)
        feature_5_5 = tf.layers.conv2d_transpose(feature_5, NUM_OF_CLASSESS, (32, 32), strides=16, padding="same")
        print("feature_5_logits.shape", feature_5_5.shape)
        merge_all = tf.concat(values=[feature_1_1, feature_2_2, feature_3_3, feature_4_4, feature_5_5, conv_t2_4], axis=3, name="merge_all")
        merge_all_1 = tf.layers.conv2d(merge_all, NUM_OF_CLASSESS, (1, 1), activation="relu", padding="same")
        annotation_pred_4 = tf.argmax(merge_all_1, dimension=3, name="prediction", output_type=tf.int32)
    return tf.expand_dims(annotation_pred_4, dim=3), merge_all_1
def inference_feature_pary(image,  keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)
    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = np.squeeze(model_data['layers'])
    processed_image = utils.process_image(image, mean_pixel)
    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        pool5 = tf.layers.max_pooling2d(image_net["conv5_3"], 2, 2)
        relu6 = tf.layers.conv2d(pool5, 4096, (7, 7), activation="relu",padding="same", name="relu6")
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)
        relu7 = tf.layers.conv2d(relu_dropout6, 4096, (1, 1), activation="relu", padding="same", name="relu7")
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)
        conv8 = tf.layers.conv2d(relu_dropout7, NUM_OF_CLASSESS, (1, 1), padding="same", name="conv8")
        print("conv8.shape", conv8.shape)
        # 卷积第一次
        deconv_shape1 = image_net["pool4"].get_shape()
        # image_1 = tf.image.resize_images(image, (deconv_shape1[1].value, deconv_shape1[2].value))
        conv_t1 = tf.layers.conv2d_transpose(conv8, deconv_shape1[3].value, (4, 4), strides=2, padding="same")
        print("conv_t1.shape:", conv_t1.shape)
        # fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")
        fuse_1 = tf.concat(values=[conv_t1, image_net["pool4"]], axis=3, name="fuse_1")
        print("fuse_1.shape", fuse_1.shape)
        fuse_1 = tf.layers.conv2d(fuse_1, deconv_shape1[3].value,  (3, 3), activation="relu",padding="same")
        fuse_1 = tf.layers.conv2d(fuse_1, deconv_shape1[3].value, (3, 3), activation="relu", padding="same")
        print("fuse_1_conv.shape", fuse_1.shape)
        # 反卷积16s
        # conv_t2_16 = tf.layers.conv2d_transpose(fuse_1, NUM_OF_CLASSESS, (32, 32), strides=16, padding="same")
        # print("conv_t2_16.shape:", conv_t2_16.shape)
        # annotation_pred_16 = tf.argmax(conv_t2_16, dimension=3, name="prediction", output_type=tf.int32)
        #卷积第二次
        deconv_shape2 = image_net["pool3"].get_shape()
        # image_2 = tf.image.resize_images(image, (deconv_shape2[1].value, deconv_shape2[2].value))
        conv_t2 = tf.layers.conv2d_transpose(fuse_1, deconv_shape2[3].value, (4, 4), strides=2, padding="same")
        fuse_2 = tf.concat(values=[conv_t2, image_net["pool3"]], axis=3, name="fuse_2")
        print("fuse_2.shape", fuse_2.shape)
        fuse_2 = tf.layers.conv2d(fuse_2, deconv_shape2[3].value, (3, 3), activation="relu", padding="same")
        fuse_2 = tf.layers.conv2d(fuse_2, deconv_shape2[3].value, (3, 3), activation="relu", padding="same")
        print("fuse_2_conv.shape", fuse_2.shape)
    #反卷积 8s
        # conv_t2_8 = tf.layers.conv2d_transpose(fuse_2, NUM_OF_CLASSESS, (16, 16), strides=8, padding="same")
        # print("conv_t2_8.shape:", conv_t2_8.shape)
        # annotation_pred_8 = tf.argmax(conv_t2_8, dimension=3, name="prediction", output_type=tf.int32)
        #卷积第三次
        deconv_shape3 = image_net["pool2"].get_shape()
        # image_3 = tf.image.resize_images(image, (deconv_shape3[1].value, deconv_shape3[2].value))
        conv_t3 = tf.layers.conv2d_transpose(fuse_2, deconv_shape3[3].value, (4, 4), strides=2, padding="same")
        fuse_3 = tf.concat(values=[conv_t3, image_net["pool2"]], axis=3, name="fuse_3")
        print("fuse_3.shape:", fuse_3.shape)
        fuse_3 = tf.layers.conv2d(fuse_3, deconv_shape3[3].value, (3, 3), activation="relu", padding="same")
        fuse_3 = tf.layers.conv2d(fuse_3, deconv_shape3[3].value, (3, 3), activation="relu", padding="same")
        print("fuse_3_conv.shape:", fuse_3.shape)
        conv_t2_4 = tf.layers.conv2d_transpose(fuse_3, NUM_OF_CLASSESS, (8, 8), strides=4, padding="same")
        print("conv_t2_4.shape:", conv_t2_4.shape)
        feature_1 = image_net['relu1_2']
        print("feature_1.shape", feature_1.shape)
        feature_1_1 = tf.layers.conv2d(feature_1, NUM_OF_CLASSESS, (1, 1), activation="relu", padding="same")

        print("feature_1_logits.shape", feature_1_1.shape)
        feature_2 = image_net['relu2_2']
        feature_shape_2 = image_net['relu2_2'].get_shape()
        feature_2_1 = tf.layers.conv2d_transpose(feature_1, feature_shape_2[3].value, (2, 2), strides=2, padding="same")
        print("123456789", feature_2_1.shape)

        feature_2 = tf.add(feature_2, feature_2_1)
        print("feature_2.shape", feature_2.shape)
        feature_2_2 = tf.layers.conv2d_transpose(feature_2, NUM_OF_CLASSESS, (4, 4), strides=2, padding="same")
        print("feature_2_logits.shape", feature_2_2.shape)
        feature_3 = image_net['relu3_4']
        print("feature_3.shape", feature_3.shape)
        feature_shape_3 = image_net['relu3_4'].get_shape()
        feature_3_1 = tf.layers.conv2d_transpose(feature_2, feature_shape_3[3].value, (2, 2), strides=2, padding="same")
        feature_3 = tf.add(feature_3_1,feature_3 )
        feature_3_3 = tf.layers.conv2d_transpose(feature_3, NUM_OF_CLASSESS, (8, 8), strides=4, padding="same")
        print("feature_3_logits.shape", feature_3_3.shape)
        feature_4 = image_net['relu4_4']
        print("feature_4.shape", feature_4.shape)
        feature_shape_4 = image_net['relu4_4'].get_shape()
        feature_4_1 = tf.layers.conv2d_transpose(feature_3, feature_shape_4[3].value, (2, 2), strides=2, padding="same")
        feature_4 = tf.add(feature_4_1, feature_4)

        feature_4_4 = tf.layers.conv2d_transpose(feature_4, NUM_OF_CLASSESS, (16, 16), strides=8, padding="same")
        print("feature_4_logits.shape", feature_4_4.shape)
        feature_5 = image_net['conv5_3']
        print("feature_5.shape", feature_5.shape)
        feature_shape_5 = image_net['conv5_3'].get_shape()
        feature_5_1 = tf.layers.conv2d_transpose(feature_4, feature_shape_5[3].value, (2, 2), strides=2, padding="same")
        feature_5 = tf.add(feature_5_1, feature_5)

        feature_5_5 = tf.layers.conv2d_transpose(feature_5, NUM_OF_CLASSESS, (32, 32), strides=16, padding="same")
        print("feature_5_logits.shape", feature_5_5.shape)
        merge_all = tf.concat(values=[feature_1_1, feature_2_2, feature_3_3, feature_4_4, feature_5_5, conv_t2_4], axis=3, name="merge_all")
        merge_all_1 = tf.layers.conv2d(merge_all, NUM_OF_CLASSESS, (1, 1), activation="relu", padding="same")
        annotation_pred_4 = tf.argmax(merge_all_1, dimension=3, name="prediction", output_type=tf.int32)

    return tf.expand_dims(annotation_pred_4, dim=3), merge_all_1
def inference_feature_pymaid(image, binary_image,  keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)
    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = np.squeeze(model_data['layers'])
    processed_image = utils.process_image(image, mean_pixel)
    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        binary_image = tf.layers.conv2d(binary_image, filters=3, kernel_size=(1, 1),
                                         strides=1, padding='same',
                                         activation=None)
        pool5 = tf.layers.max_pooling2d(image_net["conv5_3"], 2, 2)
        relu6 = tf.layers.conv2d(pool5, 4096, (7, 7), activation="relu",padding="same", name="relu6")
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)
        relu7 = tf.layers.conv2d(relu_dropout6, 4096, (1, 1), activation="relu", padding="same", name="relu7")
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)
        conv8 = tf.layers.conv2d(relu_dropout7, NUM_OF_CLASSESS, (1, 1), padding="same", name="conv8")
        print("conv8.shape", conv8.shape)
        # 卷积第一次
        deconv_shape1 = image_net["pool4"].get_shape()
        conv_t1 = tf.layers.conv2d_transpose(conv8, deconv_shape1[3].value, (4, 4), strides=2, padding="same")

        print("conv_t1.shape:", conv_t1.shape)
        image_1 = tf.image.resize_images(image, (deconv_shape1[1].value, deconv_shape1[2].value))
        binary_image_1 = tf.image.resize_images(binary_image, (deconv_shape1[1].value, deconv_shape1[2].value))
        binary_image_1 = tf.cast(binary_image_1, tf.float32)
        print("binary_image_1.shape:", binary_image_1.shape)
        merge_1 = tf.concat([image_1, binary_image_1], axis=3)
        guided_post_1 = tf.layers.conv2d(merge_1, filters=32, kernel_size=(1, 1),
                                           strides=1, padding='same',
                                           activation="relu")
        guided_post_1 = tf.layers.conv2d(guided_post_1, filters=deconv_shape1[3].value, kernel_size=(1, 1),
                                           strides=1, padding='same',
                                           activation=None)
        # guided_post_1_1 = tf.layers.conv2d(guided_post_1, deconv_shape1[3].value, (1, 1), padding="same")
        print("guided_post_1.shape:", guided_post_1.shape)
        conv_t1 = guided_filter(guided_post_1, conv_t1, 6, 0.01, nhwc=True)
        print("conv_t1.shape:", conv_t1.shape)
        fuse_1 = tf.concat(values=[conv_t1, image_net["pool4"]], axis=3, name="fuse_1")
        print("fuse_1.shape", fuse_1.shape)
        fuse_1 = tf.layers.conv2d(fuse_1, deconv_shape1[3].value, (3, 3), activation="relu",padding="same")
        fuse_1 = tf.layers.conv2d(fuse_1, deconv_shape1[3].value, (3, 3), activation="relu", padding="same")
        print("fuse_1_conv.shape", fuse_1.shape)
        # 反卷积16s
        # conv_t2_16 = tf.layers.conv2d_transpose(fuse_1, NUM_OF_CLASSESS, (32, 32), strides=16, padding="same")
        # print("conv_t2_16.shape:", conv_t2_16.shape)
        # annotation_pred_16 = tf.argmax(conv_t2_16, dimension=3, name="prediction", output_type=tf.int32)
        #卷积第二次
        deconv_shape2 = image_net["pool3"].get_shape()
        conv_t2 = tf.layers.conv2d_transpose(fuse_1, deconv_shape2[3].value, (4, 4), strides=2, padding="same")
        image_2 = tf.image.resize_images(image, (deconv_shape2[1].value, deconv_shape2[2].value))
        binary_image_2 = tf.image.resize_images(binary_image, (deconv_shape2[1].value, deconv_shape2[2].value))
        binary_image_2 = tf.cast(binary_image_2, tf.float32)
        print("binary_image_1.shape:", binary_image_2.shape)
        merge_2 = tf.concat([image_2, binary_image_2], axis=3)
        guided_post_2 = tf.layers.conv2d(merge_2, filters=32, kernel_size=(1, 1),
                                         strides=1, padding='same',
                                         activation="relu")
        guided_post_2 = tf.layers.conv2d(guided_post_2, filters=deconv_shape2[3].value, kernel_size=(1, 1),
                                         strides=1, padding='same',
                                         activation=None)
        # guided_post_1_1 = tf.layers.conv2d(guided_post_1, deconv_shape1[3].value, (1, 1), padding="same")
        print("guided_post_2.shape:", guided_post_2.shape)
        conv_t2 = guided_filter(guided_post_2, conv_t2, 6, 0.01, nhwc=True)
        print("conv_t1.shape:", conv_t2.shape)

        fuse_2 = tf.concat(values=[conv_t2, image_net["pool3"]], axis=3, name="fuse_2")
        print("fuse_2.shape", fuse_2.shape)
        fuse_2 = tf.layers.conv2d(fuse_2, deconv_shape2[3].value, (3, 3), activation="relu", padding="same")
        fuse_2 = tf.layers.conv2d(fuse_2, deconv_shape2[3].value, (3, 3), activation="relu", padding="same")
        print("fuse_2_conv.shape", fuse_2.shape)
         #反卷积 8s
        conv_t2_8 = tf.layers.conv2d_transpose(fuse_2, NUM_OF_CLASSESS, (16, 16), strides=8, padding="same")
        print("conv_t2_8.shape:", conv_t2_8.shape)
        annotation_pred_8 = tf.argmax(conv_t2_8, dimension=3, name="prediction", output_type=tf.int32)
        #卷积第三次
        deconv_shape3 = image_net["pool2"].get_shape()
        conv_t3 = tf.layers.conv2d_transpose(fuse_2, deconv_shape3[3].value, (4, 4), strides=2, padding="same")

        image_3 = tf.image.resize_images(image, (deconv_shape3[1].value, deconv_shape3[2].value))
        binary_image_3 = tf.image.resize_images(binary_image, (deconv_shape3[1].value, deconv_shape3[2].value))
        binary_image_3 = tf.cast(binary_image_3, tf.float32)
        print("binary_image_3.shape:", binary_image_3.shape)
        merge_3 = tf.concat([image_3, binary_image_3], axis=3)
        guided_post_3 = tf.layers.conv2d(merge_3, filters=32, kernel_size=(1, 1),
                                         strides=1, padding='same',
                                         activation="relu")
        guided_post_3 = tf.layers.conv2d(guided_post_3, filters=deconv_shape3[3].value, kernel_size=(1, 1),
                                         strides=1, padding='same',
                                         activation=None)
        # guided_post_1_1 = tf.layers.conv2d(guided_post_1, deconv_shape1[3].value, (1, 1), padding="same")
        print("guided_post_3.shape:", guided_post_3.shape)
        conv_t3 = guided_filter(guided_post_3, conv_t3, 6, 0.01, nhwc=True)
        print("conv_t3.shape:", conv_t3.shape)
        fuse_3 = tf.concat(values=[conv_t3, image_net["pool2"]], axis=3, name="fuse_3")
        print("fuse_3.shape:", fuse_3.shape)
        fuse_3 = tf.layers.conv2d(fuse_3, deconv_shape3[3].value, (3, 3), activation="relu", padding="same")
        fuse_3 = tf.layers.conv2d(fuse_3, deconv_shape3[3].value, (3, 3), activation="relu", padding="same")
        print("fuse_3_conv.shape:", fuse_3.shape)
        conv_t2_4 = tf.layers.conv2d_transpose(fuse_3, NUM_OF_CLASSESS, (8, 8), strides=4, padding="same")
        print("conv_t2_4.shape:", conv_t2_4.shape)
        annotation_pred_4 = tf.argmax(conv_t2_4, dimension=3, name="prediction", output_type=tf.int32)
        feature_1 = image_net['relu1_2']
        print("feature_1.shape", feature_1.shape)
        feature_1_1 = tf.layers.conv2d(feature_1, NUM_OF_CLASSESS, (1, 1), activation="relu", padding="same")
        print("feature_1_logits.shape", feature_1_1.shape)
        feature_2 = image_net['relu2_2']
        print("feature_2.shape", feature_2.shape)
        feature_2_2 = tf.layers.conv2d_transpose(feature_2, NUM_OF_CLASSESS, (4, 4), strides=2, padding="same")
        print("feature_2_logits.shape", feature_2_2.shape)
        feature_3 = image_net['relu3_4']
        print("feature_3.shape", feature_3.shape)
        feature_3_3 = tf.layers.conv2d_transpose(feature_3, NUM_OF_CLASSESS, (8, 8), strides=4, padding="same")
        print("feature_3_logits.shape", feature_3_3.shape)
        feature_4 = image_net['relu4_4']
        print("feature_4.shape", feature_4.shape)
        feature_4_4 = tf.layers.conv2d_transpose(feature_4, NUM_OF_CLASSESS, (16, 16), strides=8, padding="same")
        print("feature_4_logits.shape", feature_4_4.shape)
        feature_5 = image_net['conv5_3']
        print("feature_5.shape", feature_5.shape)
        feature_5_5 = tf.layers.conv2d_transpose(feature_5, NUM_OF_CLASSESS, (32, 32), strides=16, padding="same")
        print("feature_5_logits.shape", feature_5_5.shape)
        merge_all = tf.concat(values=[feature_1_1, feature_2_2, feature_3_3, feature_4_4, feature_5_5, conv_t2_4],
                              axis=3, name="merge_all")
        merge_all_1 = tf.layers.conv2d(merge_all, NUM_OF_CLASSESS, (1, 1), activation="relu", padding="same")
        annotation_pred_all = tf.argmax(merge_all_1, dimension=3, name="prediction", output_type=tf.int32)
    return tf.expand_dims(annotation_pred_all, dim=3), merge_all_1
def train(loss_val, var_list,c):
    optimizer = tf.train.AdamOptimizer(c)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)
def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    is_training = tf.placeholder(tf.bool)
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation_seg = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")
    annotation_binary = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")
    global_ = tf.Variable(tf.constant(0))
    c = tf.train.exponential_decay(FLAGS.learning_rate, global_, decay_steps, decay_rate, staircase=True)
    pre_annotation, logits = inference_yunshi_jia1_feature_paymid_conv3x3(image, keep_probability)
    annotation_binary_1 = tf.cast(annotation_binary, tf.float32)
    annotation_binary_1 = tf.layers.conv2d(annotation_binary_1, filters=3, kernel_size=(1, 1),
                                       strides=1, padding='same',
                                       )
    image_1 = tf.cast(image, tf.float32)
    binary_image_merge = tf.concat([annotation_binary_1, image_1], axis=3)
    annotation_post = tf.layers.conv2d(binary_image_merge, filters=32, kernel_size=(1, 1),
                                       strides=1, padding='same',
                                       activation="relu")
    annotation_post = tf.layers.conv2d(annotation_post, filters=17, kernel_size=(1, 1),
                                       strides=1, padding='same',
                                       activation=None)
    pre_seg_pred_postcess = guided_filter(annotation_post, logits, 7, 0.01, nhwc=True)

    raw_prediction_1_8 = tf.reshape(pre_seg_pred_postcess, [-1, NUM_OF_CLASSESS])
    label_proc_1_8 = tf.squeeze(annotation_seg, squeeze_dims=[3])
    raw_gt_1_8 = tf.reshape(label_proc_1_8, [-1, ])
    indices_1_8 = tf.squeeze(tf.where(tf.less_equal(raw_gt_1_8, NUM_OF_CLASSESS - 1)), 1)
    gt_ignore_1_8 = tf.cast(tf.gather(raw_gt_1_8, indices_1_8), tf.int32)
    prediction_ignore_1_8 = tf.gather(raw_prediction_1_8, indices_1_8)
    pred_annotation_111_8 = tf.argmax(prediction_ignore_1_8, dimension=1)
    pred_annotation_011_8 = tf.cast(pred_annotation_111_8, tf.int32)


    with tf.name_scope("metrices_1"):
        seg_acc_1_8, seg_acc_op_1_8 = tf.metrics.accuracy(predictions=pred_annotation_011_8, labels=gt_ignore_1_8)
        seg_class_acc, seg_class_acc_op = tf.metrics.mean_per_class_accuracy(predictions=pred_annotation_011_8, labels=gt_ignore_1_8,num_classes=17)
    loss_1_8 = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction_ignore_1_8,
                                                                      labels=gt_ignore_1_8,
                                                                          name="entropy")))
    loss_1 = loss_1_8

    loss_summary_all = tf.summary.scalar("entropy_all", loss_1)
    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss_1, trainable_var, c)
    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()
    print("Setting up image reader...")
    train_records, valid_records, test_records = scene_parsing.read_dataset(FLAGS.data_dir)
    print(len(train_records))
    print(len(test_records))
    print(len(valid_records))
    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)
        valid_dataset_reader = dataset.BatchDatset(valid_records, image_options)
    test_dataset_reader = dataset.BatchDatset(test_records, image_options)

    sess = tf.Session(config=config)

    print("Setting up Saver...")
    saver = tf.train.Saver(max_to_keep=5, var_list=tf.trainable_variables())

    train_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/train', sess.graph)
    valid_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/valid', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/test')
    sess.run(tf.global_variables_initializer())
    # ckpt = tf.train.latest_checkpoint(FLAGS.logs_dir)
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    if FLAGS.mode == "train":
        min_loss = 1000.0
        for itr in xrange(MAX_ITERATION):
            sess.run(tf.local_variables_initializer())
            train_images, train_annotations_seg, train_annotations_binary = train_dataset_reader.next_batch(FLAGS.batch_size)
            train_annotations_binary_1 = train_annotations_binary // 255.0
            train_annotations_seg = train_annotations_seg - 1
            # train_images = train_images // 255.0
            feed_dict = {image: train_images, annotation_seg: train_annotations_seg, annotation_binary: train_annotations_binary_1, keep_probability: 0.85, global_:itr}
            sess.run([train_op], feed_dict=feed_dict)
            if itr % 2 == 0:
                train_seg_loss, lr, summary_str = sess.run([loss_1, c, summary_op], feed_dict=feed_dict)
                seg_acc_per_op_train_1_8 = sess.run(seg_acc_op_1_8, feed_dict=feed_dict)
                seg_acc_train_1_8 = sess.run(seg_acc_1_8, feed_dict=feed_dict)
                seg_acc_per_op_train_class = sess.run(seg_class_acc_op, feed_dict=feed_dict)
                seg_acc_train_class = sess.run(seg_class_acc, feed_dict=feed_dict)
                print("Step: %d,  Train_seg_loss:%g" % (
                itr, train_seg_loss))
                print("Step: %d, Train_seg_post_acc_8:%g" % (itr, seg_acc_train_1_8))
                print("Train_seg_post_acc_class_op: " ,  seg_acc_per_op_train_class)

                print("Step: %d, Train_seg_post_acc_class:%g"% (itr, seg_acc_train_class))
                print("Step: %d, lr:%g" % (itr, lr))
                train_writer.add_summary(summary_str, itr)
                if train_seg_loss < min_loss:
                    min_loss = train_seg_loss
                    print("*******************************************########################################")
                    print(min_loss)
                    saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)
            if itr % 29 == 0:
                valid_images, valid_annotations_seg, valid_annotations_binary = valid_dataset_reader.next_batch(
                    FLAGS.batch_size)
                valid_annotations_binary_1 = valid_annotations_binary // 255.0
                valid_annotations_seg = valid_annotations_seg - 1
                feed_dict = {image: valid_images, annotation_seg: valid_annotations_seg,
                             annotation_binary: valid_annotations_binary_1,  keep_probability: 1.0, global_:itr}
                valid_seg_loss,lr_1, summary_str1 = sess.run([loss_1, c, summary_op], feed_dict=feed_dict)
                seg_acc_per_op_valid_1_8 = sess.run(seg_acc_op_1_8, feed_dict=feed_dict)
                seg_acc_valid_1_8 = sess.run(seg_acc_1_8, feed_dict=feed_dict)
                seg_acc_per_op_valid_class = sess.run(seg_class_acc_op, feed_dict=feed_dict)
                seg_acc_valid_class = sess.run(seg_class_acc, feed_dict=feed_dict)

                print("Step: %d,  valid_seg_loss:%g" % (
                    itr, valid_seg_loss))
                print("Step: %d, valid_seg_post_acc_8:%g" % (itr, seg_acc_valid_1_8))
                print("valid_seg_post_acc_class_op: ", seg_acc_per_op_valid_class)
                print("Step: %d, valid_seg_post_acc_class:%g" % (itr, seg_acc_valid_class))
                print("Step: %d, valid_lr:%g" % (itr, lr_1))
                valid_writer.add_summary(summary_str1, itr)

    elif FLAGS.mode == "test":

        test_acc_post_8 = []
        test_acc_post_class = []
        test_per_class_heng = []
        time_all = []
        for epoch in range(test_epoch):
            test_images, test_mask_seg, test_mask_binary_1 = test_dataset_reader.next_batch(1)
            sess.run(tf.local_variables_initializer())
            test_mask_binary = test_mask_binary_1 // 255.0
            test_mask_seg_1 = test_mask_seg - 1
            # test_images = test_images // 255.0
            feed_dict = {image: test_images, annotation_seg: test_mask_seg_1,
                         annotation_binary: test_mask_binary, keep_probability: 1.0}
            test_per_start_time = time.time()
            # test_binary_pred,test_seg_pred = sess.run([pre_side_all_annotation, pre_annotation],feed_dict=feed_dict)
            test_seg_pred = sess.run(pre_seg_pred_postcess, feed_dict=feed_dict)
            test_per_end_time = time.time()

            seg_acc_per_op_test_1_8 = sess.run(seg_acc_op_1_8, feed_dict=feed_dict)
            seg_acc_test_1_8 = sess.run(seg_acc_1_8, feed_dict=feed_dict)
            seg_acc_per_op_test_class = sess.run(seg_class_acc_op, feed_dict=feed_dict)
            seg_acc_test_class = sess.run(seg_class_acc, feed_dict=feed_dict)

            print("Step: %d, test_seg_post_acc_8:%g" % (epoch, seg_acc_test_1_8))
            print("test_seg_post_acc_class_op: ", seg_acc_per_op_test_class)
            print("Step: %d, test_seg_post_acc_class:%g" % (epoch, seg_acc_test_class))

            test_acc_post_8.append(seg_acc_test_1_8)
            test_acc_post_class.append(seg_acc_test_class)
            test_per_class_heng.append(seg_acc_per_op_test_class)
            time_all.append(test_per_end_time - test_per_start_time)

            pred_seg_visal_post_8 = np.argmax(test_seg_pred, axis=3)

            for itr in range(1):
                vis.saveResult("./test_predict_seg_post", pred_seg_visal_post_8[itr].astype(np.uint8), True, NUM_OF_CLASSESS, epoch)
                # vis.saveResult("./test_predict_seg_post_8", pred_seg_visal_post_8[itr].astype(np.uint8), True,
                #                NUM_OF_CLASSESS, epoch)
                # vis.saveResult("./test_predict_seg_post_16", pred_seg_visal_post_16[itr].astype(np.uint8), True,
                #                NUM_OF_CLASSESS, epoch)
                # vis.saveResult("./test_predict_seg_post_all", pred_seg_visal_post_all[itr].astype(np.uint8), True,
                #                NUM_OF_CLASSESS, epoch)
        print("每张图像平均时间：", np.mean(time_all))
        print("测试时滤波后图像平均准确率：", np.mean(test_acc_post_8))
        print("测试时滤波后图像每类平均准确率：", np.mean(test_acc_post_class))
        print("测试时滤波后图像每类平均准确率,横向对比：", np.mean(test_per_class_heng, axis=0))
        # print("测试时滤波后图像平均准确率_16：", np.mean(test_acc_post_16))
        # print("测试时滤波后图像平均准确率_all：", np.mean(test_acc_post_all))
if __name__ == "__main__":
    tf.app.run()
