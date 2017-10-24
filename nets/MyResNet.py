import tensorflow as tf
import collections

slim = tf.contrib.slim

Block = collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])

def subsample(inputs, factor, scope=None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)

def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               outputs_collections='collections',
               scope=None):

    with tf.variable_scope(scope) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        if depth==depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:

            shortcut = slim.conv2d(preact, depth, [1, 1],
                                   stride=stride, normalizer_fn=None,
                                   activation_fn=None, scope='shortcut')
        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')

        residual = slim.conv2d(residual, depth_bottleneck, [3, 3], stride=stride, padding='SAME', scope='conv2')

        residual = slim.conv2d(residual, depth, [1, 1], stride=1, scope='conv3')

        output = shortcut+residual

        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)



def resnet_50(input):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    net = input
    net = slim.conv2d(net, 64, 7, stride=2, scope='conv1', padding='SAME')
    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
    with tf.variable_scope('resnet_50'):
        for i, block in enumerate(blocks):
            with tf.variable_scope(block.scope):
                args = block.args
                for j, arg in enumerate(args):
                    depth, depth_bottleneck, stride = arg
                    net = bottleneck(net, depth, depth_bottleneck, stride, scope='unit_'+str(j))
    endpoints = slim.utils.convert_collection_to_dict('collections')
    return net, endpoints
'''
def test():
    inputs = tf.Variable(tf.truncated_normal((4, 512, 512, 3), stddev=0.01))

    endpoints = bottleneck(inputs, 256, 64, stride=2, scope='ResNet')



    net = resnet_50(inputs)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        ret = sess.run(net)
        print(ret)
        print(1)

        endpoints = slim.utils.convert_collection_to_dict('collections')
        print(endpoints)

test()
'''