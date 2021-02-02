''' code for Resnet Model'''
import tensorflow as tf

from .res_block import Res_Block2, Res_Block1

class ResNet(tf.keras.Model):
    """used to Create the different  Resnet model with the given configuration

    Args:
        a (int): multiple  resnetblock with multiple of a
        b (int): multiple  resnetblock with multiple of b
        c (int): multiple  resnetblock with multiple of c
        d (int): multiple  resnetblock with multiple of d
        num_layers (int): number of layers in Resnet with only 5 choices [18,34,50,101,151]
        num_classes (int):  number of classes in the output layer

    """
    def __init__(self,a,b,c,d,num_layers,num_classes):
        super(ResNet,self).__init__()
        self.conv = tf.keras.layers.Conv2D(64,7,padding='same')
        self.bn   = tf.keras.layers.BatchNormalization()
        self.act  = tf.keras.layers.Activation('relu')
        self.max_pool = tf.keras.layers.MaxPool2D((3,3))
        if num_layers < 50:
            self.resa = Res_Block1(64,a)
            self.resb = Res_Block1(128,b)
            self.resc = Res_Block1(256,c)
            self.resd = Res_Block1(512,d)
        else:
            self.resa = Res_Block2(64,a)
            self.resb = Res_Block2(128,b)
            self.resc = Res_Block2(256,c)
            self.resd = Res_Block2(512,d)

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        if num_classes > 2:
            self.classifier = tf.keras.layers.Dense(num_classes,activation ='softmax')
        else:
            self.classifier = tf.keras.layers.Dense(1,activation ='sigmoid')

    def call(self,inputs):
        x=self.conv(inputs)
        x=self.bn(x)
        x=self.act(x)
        x=self.max_pool(x)

        x=self.resa(x)
        x=self.resb(x)
        x=self.resc(x)
        x=self.resd(x)

        x=self.global_pool(x)
        return self.classifier(x)

def ResNet18(num_classes):
    """ResNet18 model

    Args:
        num_classes (int): number of classes in the output layer

    Returns:
        [model]: Resnet18 model
    """
    return ResNet(2,2,2,2,18,num_classes)


def ResNet34(num_classes):
    """ResNet34 model

    Args:
        num_classes (int): number of classes in the output layer

    Returns:
        [model]: Resnet34 model
    """
    return ResNet(3,4,6,3,34,num_classes)


def ResNet50(num_classes):
    """ResNet50 model

    Args:
        num_classes (int): number of classes in the output layer

    Returns:
        [model]: Resnet50 model
    """
    return ResNet(3,4,6,3,50,num_classes)


def ResNet101(num_classes):
    """ResNet101 model

    Args:
        num_classes (int): number of classes in the output layer

    Returns:
        [model]: Resnet101 model
    """
    return ResNet(3,4,23,3,101,num_classes)

def ResNet151(num_classes):
    """ResNet151 model

    Args:
        num_classes (int): number of classes in the output layer

    Returns:
        [model]: Resnet151 model
    """
    return ResNet(3,8,36,3,151,num_classes)

