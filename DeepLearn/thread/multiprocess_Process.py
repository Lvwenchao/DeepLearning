# write by Mrlv
# coding:utf-8
import time
import multiprocessing
from multiprocessing import Process


def get_html(times):
    time.sleep(times)
    url = "get html success: https://www.tread.com/{}".format(times)
    return url


if __name__ == '__main__':
    precess1 = Process(target=get_html, args=(1,))
    precess1.start()
    print(precess1.pid)
    precess1.join()
    print("main process end")

    # 线程池的使用
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    # result = pool.apply_async(get_html, args=(2,))
    # pool.close()
    # pool.join()
    # print(result.get())

    for result in pool.imap(get_html, [2, 3, 4]):
        print(result)
