# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2021/5/13 15:39
# @FileName : test_loadData.py
# @Software : PyCharm
from tools import loadData

dataset = loadData.DataSets()
ROOT = r"G:\编程\深度学习与TF-PPT和代码\dataset\pokemon"


def test_load_dataset():
    name_label = dataset.load_dataset_label(ROOT)
    print(name_label)


def test_load_csv():
    name_label = dataset.load_dataset_label(ROOT)
    dataset.load_csv(ROOT, name_label, )
