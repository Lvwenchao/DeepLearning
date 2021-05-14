# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2021/5/14 10:04
# @FileName : dataset.py
# @Software : PyCharm
import csv
import tensorflow as tf
import os
import glob
import random
from sklearn.model_selection import train_test_split

"""
 自定义数据集
"""


def load_dataset_label(root, mode='train'):
    """

    :param root: 图片文件夹上一路径
    :param mode:
    :return:
    """
    name_label = {}
    for name in sorted(os.listdir(os.path.join(root))):
        # 如果不是文件夹跳过
        if not os.path.isdir(os.path.join(root)):
            continue
        name_label[name] = len(name_label.keys())
    return name_label


def load_csv(root, filename, name_label):
    images = []
    # 加载所有图像路径到列表中
    # glob.glob 返回当前路径下的所有文件路径
    if not os.path.exists(os.path.join(root, filename)):
        image_paths = []
        for name in name_label.keys():
            image_paths += glob.glob(os.path.join(root, name, "*.jpg"))
            image_paths += glob.glob(os.path.join(root, name, "*.png"))
            image_paths += glob.glob(os.path.join(root, name, "*.jpeg"))
            image_paths += glob.glob(os.path.join(root, name, "*.tiff"))

        print("image_number:{}".format(len(image_paths)))
        random.shuffle(image_paths)
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            for image_path in image_paths:
                name = image_path.split(os.sep)[-2]
                label = name_label[name]
                writer.writerow([image_path, label])


def preprocess(x, y):
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3)  # 图片解码
    x = tf.image.resize(x, [244, 244])  # 图片缩放

    # 数据增强
    x = tf.image.random_flip_left_right(x)  # 左右镜像
    x = tf.image.random_crop(x, [224, 224, 3])  # 随机裁剪

    # 归一化
    x = tf.cast(x, dtype=tf.float32) / 255.

    # 标准化
    img_mean = tf.constant([0.485, 0.456, 0.406])
    img_std = tf.constant([0.229, 0.224, 0.225])
    # y = tf.convert_to_tensor(y, dtype=tf.int32)

    x = (x - img_mean) / img_std
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    return x, y


def read_csv(filename, test_split=0.3):
    image_paths, labels = [], []
    with open(os.path.join(filename)) as f:
        reader = csv.reader(f)
        for row in reader:
            image_path, label = row
            image_paths.append(image_path)
            labels.append(label)

    image_train, image_test, label_train, label_test = train_test_split(image_paths,
                                                                        labels,
                                                                        test_size=test_split,
                                                                        random_state=42)
    print(len(image_test), len(label_test))
    print(len(image_train), len(label_train))
    db_train = tf.data.Dataset.from_tensor_slices((image_train, label_train))
    db_train = db_train.shuffle(1000).map(preprocess).batch(32)
    db_test = tf.data.Dataset.from_tensor_slices((image_test, label_test))
    db_test = db_test.shuffle(1000).map(preprocess).batch(32)

    return db_train, db_test

