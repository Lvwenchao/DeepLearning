# write by Mrlv
# coding:utf-8
import tensorflow as tf
import os
import cv2
from tensorflow.keras import optimizers
from tools import loadData, models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# readData
def main():
    batch_size = 128
    train_data, test_data = loadData.minist_data(batch_size)
    simple = next(iter(train_data))
    print(simple[0].shape, simple[1].shape)

    # 导入模型
    network = models.MyModel(784)

    network.compile(optimizer=optimizers.Adam(lr=0.01),
                    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    network.load_weights(r'E:\pythonProject\DeepLearn\resources\models\mnist').expect_partial()

    # 对存储的模型进行测试
    network.evaluate(test_data)

    # 预测
    img = simple[0][0]
    pre = network(img)
    pre_label = tf.argmax(pre, axis=1)
    print(img.shape)
    print(pre_label)

    # show predict img
    cv2.namedWindow('img', 0)
    cv2.imshow('img', img.numpy())
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
