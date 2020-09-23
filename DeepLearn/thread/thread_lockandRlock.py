# write by Mrlv
# coding:utf-8
import threading

count = 0

lock = threading.RLock()


# 数据加
def add():
    global count
    global lock
    for i in range(1000000):
        lock.acquire()
        lock.acquire()
        count += 1
        lock.release()
        lock.release()


# 数据减
def dec():
    global count
    global lock
    for i in range(1000000):
        lock.acquire()
        count -= 1
        lock.release()


def main():
    add_thread = threading.Thread(target=add)
    dec_thread = threading.Thread(target=dec)
    add_thread.start()
    dec_thread.start()
    add_thread.join()
    dec_thread.join()
    print(count)


if __name__ == '__main__':
    main()
