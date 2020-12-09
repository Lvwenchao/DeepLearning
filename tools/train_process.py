# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2020/11/2 20:17
# @FileName : train.py
# @Software : PyCharm
import tensorflow as tf
import numpy as np


def loss(logit, pre):
    logit = tf.argmax(logit, axis=-1)
    ce_loss = tf.losses.categorical_crossentropy(logit, pre, from_logits=True)
    ce_loss = tf.reduce_mean(ce_loss)
    return ce_loss


def evl_num(logit, pre):
    logit = tf.argmax(logit, axis=-1)
    logit = tf.cast(logit, dtype=tf.int64)
    collect = tf.cast(tf.equal(logit, pre), dtype=tf.int32)
    return collect




