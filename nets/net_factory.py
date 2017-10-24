import vgg
import resnet_v1
import resnet_v2
net_dict = {
    "vgg": vgg,
    "resnet_v1": resnet_v1,
    "resnet_v2": resnet_v2
}

def get_basenet(name, inputs):

    net = net_dict[name];
    return net.basenet(inputs);
