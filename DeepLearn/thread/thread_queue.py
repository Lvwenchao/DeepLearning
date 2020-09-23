# write by Mrlv
# coding:utf-8
import threading
import time
from queue import Queue

count = 0


# 自定义多线程集合
class MyThread(threading.Thread):
    def __init__(self, lock, threadName):
        super(MyThread, self).__init__(name=threadName)
        self.lock = lock

    def run(self):
        global count
        self.lock.acquire()
        for i in range(100):
            count += 1
            time.sleep(0.3)
            print(self.getName(), count)
        self.lock.release()


# 获取页面内容
def get_detail_html(queue):
    while True:
        url = queue.get()
        print("开始获取内容详情:{}".format(url))
        time.sleep(0.3)
        print("获取内容详情结束")


# 获取页面链接
def get_detail_url(queue):
    while True:
        print("开始获取详情页链接")
        time.sleep(0.3)
        for i in range(20):
            queue.put("https://www.thread.com//" + str(i))
        print("获取详情页链接结束")


def main():
    # 共享变量Queue
    queue = Queue(200)
    get_html_thread = threading.Thread(target=get_detail_html, args=(queue,))
    get_ulr_thread = threading.Thread(target=get_detail_url, args=(queue,))

    # 将线程变为守护线程
    # get_html_thread.setDaemon(True)
    # get_ulr_thread.setDaemon(True)
    startime = time.time()
    get_ulr_thread.start()
    get_html_thread.start()

    # 线程阻
    get_html_thread.join()
    get_ulr_thread.join()

    print("last time:{}".format(time.time() - startime))


if __name__ == '__main__':
    main()
