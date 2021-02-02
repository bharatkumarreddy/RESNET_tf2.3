'''  code for resnet blocks'''

import tensorflow as tf

class Res_Block1(tf.keras.Model):
    """type 1 Resnet block used in the models which have less than 50 layers

    Args:
        filters (int): no of filters in the conv layers
        multiple (int): number of times the block is repeated
    """
    def __init__(self,filters,multiple):
        super(Res_Block1,self).__init__()
        self.res_block=tf.keras.Sequential()
        for _ in range(multiple):
            self.res_block.add(tf.keras.layers.Conv2D(filters,3,padding='same'))
            self.res_block.add(tf.keras.layers.BatchNormalization())
            self.res_block.add(tf.keras.layers.Activation('relu'))

        self.down = tf.keras.layers.Conv2D(filters,3,padding='same')
        self.act   = tf.keras.layers.Activation('relu')
        self.add   = tf.keras.layers.Add()
        self.max_pool = tf.keras.layers.MaxPool2D()

    def call(self,input_tensor):
        x=self.res_block(input_tensor)
        y=self.down(input_tensor)
        x=self.add([x,y])
        x=self.act(x)
        x=self.max_pool(x)
        return x

class Res_Block2(tf.keras.Model):
    """type 2 Resnet block used in the models which have greater than 50 layers

    Args:
        filters (int): no of filters in the conv layers
        multiple (int): number of times the block is repeated
    """
    def __init__(self,filters,multiple):
        super(Res_Block2,self).__init__()
        self.res_block=tf.keras.Sequential()
        for _ in range(multiple):
            self.res_block.add(tf.keras.layers.Conv2D(filters,1,padding='same',activation='relu'))
            self.res_block.add(tf.keras.layers.Conv2D(filters,3,padding='same',activation='relu'))
            self.res_block.add(tf.keras.layers.Conv2D(4*filters,1,padding='same',activation='relu'))
            self.res_block.add(tf.keras.layers.BatchNormalization())
        self.down = tf.keras.layers.Conv2D(4*filters,3,padding='same')
        self.act   = tf.keras.layers.Activation('relu')
        self.add   = tf.keras.layers.Add()
        self.max_pool = tf.keras.layers.MaxPool2D()

    def call(self,input_tensor):
        x=self.res_block(input_tensor)
        y=self.down(input_tensor)
        x=self.add([x,y])
        x=self.act(x)
        x=self.max_pool(x)
        return x
