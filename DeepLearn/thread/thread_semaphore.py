# write by Mrlv
# coding:utf-8
import threading
import time


class GetHtmlContent(threading.Thread):
    def __init__(self, sem, url):
        super(GetHtmlContent, self).__init__()
        self.sem = sem
        self.url = url

    def run(self):
        time.sleep(2)
        print("get content success:{}".format(self.url))
        self.sem.release()


class CollectionUrl(threading.Thread):
    def __init__(self, sem):
        self.sem = sem
        super(CollectionUrl, self).__init__()

    def run(self):
        for i in range(20):
            self.sem.acquire()
            ghc = GetHtmlContent(self.sem, "https://www.thread.com/{}".format(i))
            ghc.start()


sem = threading.Semaphore(3)
cu = CollectionUrl(sem)
cu.start()
