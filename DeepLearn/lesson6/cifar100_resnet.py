# write by Mrlv
# coding:utf-8
import tensorflow as tf
from tensorflow.keras import datasets, optimizers
from DeepLearn.tools.preprocess import preprocess
from DeepLearn.tools.customize import resnet18

batch_size = 30
(x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
y_train = tf.squeeze(y_train)
y_test = tf.squeeze(y_test)
print("data:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

train_date = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
train_date = train_date.shuffle(10000).map(preprocess).batch(batch_size)
test_data = test_data.shuffle(10000).map(preprocess).batch(batch_size)

sample = next(iter(train_date))
print(sample[0].shape, sample[1].shape, tf.reduce_max(sample[1]), tf.reduce_min(sample[1]))


def main():
    model = resnet18()
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()
    optimizer = optimizers.Adam(lr=1e-3)

    for epoch in range(50):
        for step, (x_train, y_train) in enumerate(train_date):
            # [b,32,32,3]->[b,1,1,52]
            with tf.GradientTape() as tape:
                # [b,32,32,3]->[b,100]
                logits = model(x_train)
                y_onehot = tf.one_hot(y_train, depth=100)

                # compute loss
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(step, "loss:", float(loss))

        # test
        correct_total, num_total = 0, 0
        for step, (x, y) in enumerate(test_data):
            # [b,32,32,3}->[b,1,1,512]
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            # [b,100]->[b,]
            pre = tf.argmax(prob, axis=1)
            pre = tf.cast(pre, dtype=tf.int32)
            # compute accrucy
            correct = tf.cast(tf.equal(y, pre), dtype=tf.int32)
            correct = tf.reduce_sum(correct)
            correct_total += int(correct)
            num_total += x.shape[0]

        accuracy = float(correct_total / num_total)
        print(epoch, "acc", accuracy)


if __name__ == '__main__':
    main()
