# write by Mrlv
# coding:utf-8
import multiprocessing
import time


def get_data(pipe):
    time.sleep(2)
    data = pipe.recv()
    print(data)


def produce_data(pipe):
    pipe.send('Mrlv')
    time.sleep(2)


def add_data(p_dict, key, value):
    p_dict[key] = value


def main():
    receive_pipe, send_pipe = multiprocessing.Pipe()
    progress__dict = multiprocessing.Manager().dict()
    produce = multiprocessing.Process(target=produce_data, args=(send_pipe,))
    get = multiprocessing.Process(target=get_data, args=(receive_pipe,))
    addData1 = multiprocessing.Process(target=add_data, args=(progress__dict, 'Mrlv', 21))
    addData2 = multiprocessing.Process(target=add_data, args=(progress__dict, 'Mrlv2', 22))
    produce.start()
    get.start()
    produce.join()
    get.join()
    addData1.start()
    addData2.start()
    addData1.join()
    addData2.join()
    print(progress__dict)


if __name__ == '__main__':
    main()
