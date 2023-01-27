import datetime as dt
from multiprocessing import Process, current_process, Manager
import sys

lis = [0,0,0,0,0,0,0,0]

def f(ind, lis):
    lis[str(ind)].append(1)
    print(lis)
    sys.stdout.flush()

if __name__ == '__main__':
    manager = Manager()
    lis = manager.dict({'0':manager.list([0,0,0,0,0,0,0,0]), '1':manager.list([0,0,0,0,0,0,0,0])})
    worker_count = 2
    worker_pool = []
    for x in range(worker_count):
        p = Process(target=f, args=(x,lis))
        p.start()
        worker_pool.append(p)
    for p in worker_pool:
        p.join()  # Wait for all of the workers to finish.

    # Allow time to view results before program terminates.
    a = input("Finished")  # raw_input(...) in Python 2.
    print(lis)
    print(lis['0'])