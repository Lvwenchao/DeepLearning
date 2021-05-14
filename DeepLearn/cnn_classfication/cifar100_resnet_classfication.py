# write by Mrlv
# coding:utf-8
import tensorflow as tf
from tensorflow.keras import optimizers
from tools import resnet18
from tools import DataSets

batch_size = 30
# loadData
train_data, val_data, test_data = DataSets.load_tensorflow_data(dataset_name="cifar10", batch_size=30, classes=10)
sample = next(iter(train_data))

print(sample[0].shape, sample[1].shape, tf.reduce_max(sample[1]), tf.reduce_min(sample[1]))


def main():
    model = resnet18(10)
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()
    optimizer = optimizers.Adam(lr=0.0001)
    model.compile(optimizer=optimizer,
                  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_data, epochs=10, validation_data=val_data, validation_freq=1)

    model.evaluate(test_data)
    model.save_weights(r"E:\pythonProject\DeepLearning\resources\models\cifar100"
                       r"\cifar100_resnet")

    # for epoch in range(10):
    #     for step,(x,y) in enumerate(train_data):


if __name__ == '__main__':
    main()
