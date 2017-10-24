import tensorflow as tf

from MyResNet import *
slim = tf.contrib.slim

def basenet2(inputs):
    logit, endpoints =resnet_50(inputs)
    endpoints['conv4_3'] = endpoints['vgg/resnet_50/block2/unit_2']
    endpoints['fc7'] = logit # Maybe relu should be added on the logit: relu(logit)
    return endpoints['fc7'], endpoints

def basenet_2(inputs):

    """
    backbone net of vgg16
    """
    # End_points collect relevant activations for external use.
    end_points = {}
    # Original VGG-16 blocks.
    with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME'):
        net = slim.conv2d(inputs, 128, [3, 3])
        net = slim.max_pool2d(net, [2, 2], stride=2)

        net = slim.conv2d(net, 256, [3, 3])
        net = slim.max_pool2d(net, [2, 2], stride=2)

        net = slim.conv2d(net, 512, [3, 3])
        net = slim.max_pool2d(net, [2, 2], stride=2)
        end_points['conv4_3'] = net
        net = slim.conv2d(net, 1024, [3, 3])
        net = slim.max_pool2d(net, [2, 2], stride=2)
        end_points['fc7'] = net
        net = slim.max_pool2d(net, [2, 2], stride=2)
    return net, end_points;

def basenet(inputs):
    """
    backbone net of vgg16
    """
    # End_points collect relevant activations for external use.
    end_points = {}
    # Original VGG-16 blocks.
    with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME'):
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        end_points['conv1_2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        end_points['conv2_2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        # Block 3.
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        end_points['conv3_3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        # Block 4.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        end_points['conv4_3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        # Block 5.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        end_points['conv5_3'] = net
        net = slim.max_pool2d(net, [3, 3], 1, scope='pool5')

        # fc6 as conv, dilation is added
        net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='fc6')
        end_points['fc6'] = net

        # fc7 as conv
        net = slim.conv2d(net, 1024, [1, 1], scope='fc7')
        end_points['fc7'] = net

    return net, end_points;    


def basenet_v2(inputs):

    logit ,endpoints= resnet_v2_50(inputs)
    #print(endpoints.keys())
    net =endpoints['vgg/resnet_v2_50/block3/unit_5/bottleneck_v2/conv3']
    # fc7=vgg/resnet_v2_50/block3/unit_5/bottleneck_v2/conv3  conv4_3='vgg/resnet_v2_50/block2/unit_3/bottleneck_v2/conv3'
    # (1, 32, 32, 1024) (1,64,64,512)
    endpoints['fc7'] =net
    endpoints['conv4_3'] = endpoints['vgg/resnet_v2_50/block2/unit_3/bottleneck_v2/conv3']
    return net, endpoints

def basenet_v3(inputs):

      logits, endpoints = resnet_v1_50(inputs)
      endpoints['conv4_3'] = endpoints['vgg/resnet_v1_50/block2/unit_3/bottleneck_v1/conv3']
      endpoints['fc7'] = endpoints['vgg/resnet_v1_50/block3/unit_5/bottleneck_v1/conv3']
      net = endpoints['fc7']
      return net, endpoints

