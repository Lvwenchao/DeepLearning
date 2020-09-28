# write by Mrlv
# coding:utf-8
import multiprocessing
import time


def get_html(queue):
    while True:
        url = queue.get()
        print("开始获取内容详情:{}".format(url))
        time.sleep(0.3)
        print("获取内容详情结束")


def produce_url(queue):
    while True:
        print("开始获取详情页链接")
        for i in range(20):
            queue.put("https://www.thread.com//" + str(i))
            time.sleep(0.2)
        print("获取详情页链接结束")


def main():
    cpu_count = multiprocessing.cpu_count()
    print(cpu_count)
    pool = multiprocessing.Pool(cpu_count)
    queue = multiprocessing.Manager().Queue(10)
    process1 = pool.apply_async(produce_url, args=(queue,))
    process2 = pool.apply_async(get_html, args=(queue,))
    # pro_url = multiprocessing.Process(target=produce_url, args=(queue,))
    # ger_url = multiprocessing.Process(target=get_html, args=(queue,))
    # pro_url.start()
    # ger_url.start()
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
