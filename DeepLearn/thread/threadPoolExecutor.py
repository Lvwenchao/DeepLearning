# write by Mrlv
# coding:utf-8
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


def get_html(times):
    time.sleep(times)
    url = "get html success: https://www.tread.com/{}".format(times)
    return url


def main():
    excutor = ThreadPoolExecutor(max_workers=2)
    task1 = excutor.submit(get_html, 2)
    task2 = excutor.submit(get_html, 3)

    print(task1.done(), task2.done())
    print(task1.result(), task2.result())
    time.sleep(4)
    print(task1.done(), task2.done())

    urls = [2, 3, 4, 5]
    tasks = [excutor.submit(get_html, url) for url in urls]
    for i in as_completed(tasks):
        result = i.result()
        print(result)


if __name__ == '__main__':
    main()
