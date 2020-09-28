# write by Mrlv
# coding:utf-8
# 多进程编程
# 进行多进程与多线程进行比较
import time
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

pid = os.fork()
# 当pid为0的时候存在主进程和子进程


# 斐波拉契数列
def feb(n):
    if n <= 2:
        return 1
    return feb(n - 1) + feb(n - 2)


def main1():
    with ThreadPoolExecutor(3) as executor:
        all_task = [executor.submit(feb, num) for num in range(26, 40)]
        starttime = time.time()
        for future in as_completed(all_task):
            result = future.result()
            print('task return {}'.format(result))

        endtime = time.time()
        print("多线程时间:{}".format(endtime - starttime))


def main2():
    with ProcessPoolExecutor(3) as executor:
        all_task = [executor.submit(feb, num) for num in range(26, 40)]
        starttime = time.time()
        for future in as_completed(all_task):
            result = future.result()
            print('task return {}'.format(result))

        endtime = time.time()
        print("多进程时间:{}".format(endtime - starttime))


if __name__ == '__main__':
    main1()
