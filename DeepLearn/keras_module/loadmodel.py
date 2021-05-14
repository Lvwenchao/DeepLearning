# write by Mrlv
# coding:utf-8
import tensorflow as tf
import tensorboard
import os

from tensorflow.keras import applications, layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# readData
def main():

    resnet = applications.ResNet50(weights='imagenet', include_top=False)
    resnet.build((4, 224, 224, 3))
    # resnet.summary()
    x = tf.random.normal((4, 224, 224, 3))
    out = resnet(x)

    print(out.shape)
    # 全局平均池化
    global_average_layer = layers.GlobalAveragePooling2D()
    # 利用上一层的输出作为本层的输入，测试其输出
    x = tf.random.normal([4, 7, 7, 2048])
    out = global_average_layer(x)  # 池化层降维
    print(out.shape)


if __name__ == '__main__':
    main()
