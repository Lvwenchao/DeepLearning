# write by Mrlv
# coding:utf-8
from threading import Thread, Condition
import time


# bixby助手
class Bixby(Thread):
    def __init__(self, con):
        super(Bixby, self).__init__(name='Bixby')
        self.con = con

    def run(self):
        with self.con:
            self.con.wait()
            time.sleep(1)
            print(self.name + ": 我在")
            self.con.notify()

            self.con.wait()
            time.sleep(1)
            print(self.name + ": 好啊")
            self.con.notify()

            self.con.wait()
            time.sleep(1)
            print(self.name + ": 春风拂槛露华浓")
            self.con.notify()

            self.con.wait()
            time.sleep(1)
            print(self.name + ": 会向瑶台月下逢")
            self.con.notify()


# siri
class Siri(Thread):
    def __init__(self, con):
        super(Siri, self).__init__(name='Siri')
        self.con = con

    def run(self):
        with self.con:
            print(self.name + ": Hi Bixby")
            self.con.notify()

            self.con.wait()
            time.sleep(1)
            print(self.name + ": 我们对诗吧")
            self.con.notify()

            self.con.wait()
            time.sleep(1)
            print(self.name + ": 云想衣裳花想容")
            self.con.notify()

            self.con.wait()
            time.sleep(1)
            print(self.name + ": 若非群玉山头见")
            self.con.notify()


def start():
    con = Condition()
    bixby = Bixby(con)
    siri = Siri(con)
    bixby.start()
    siri.start()


if __name__ == '__main__':
    start()
