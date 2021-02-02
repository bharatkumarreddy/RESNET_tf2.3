import tensorflow_datasets as tfds
import tensorflow as tf

from .Models import ResNet50


train=tfds.load('horses_or_humans',split='train',shuffle_files=True)
test=tfds.load('horses_or_humans',split='test',shuffle_files=True)


def preprocess(dataset):
    image=tf.image.resize(dataset['image'],(224,224))
    image=tf.cast(image,tf.float32)/255.
    label=dataset['label']
    return image,[label]

train = train.map(preprocess).batch(64)
train = train.prefetch(tf.data.experimental.AUTOTUNE)
test = test.map(preprocess).batch(64)
test = test.prefetch(tf.data.experimental.AUTOTUNE)

resnet = ResNet50(2)
resnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
resnet.fit(train,validation_data = test,epochs=100)