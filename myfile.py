from multiprocessing import Process, Queue, Lock
lock=Lock()
def f(q):
    global lock
    lock.acquire()
    print('f')
    q.put([42, None, 'hello'])
    lock.release()

def f2(q):
    global lock
    lock.acquire()
    print('f2')
    q.put(['hassan',None,'heelww'])
    lock.release()

def f3(q):
    global lock
    lock.acquire()
    print('f3')
    print(q.get())
    lock.release()

if __name__ == '__main__':
    q = Queue()
    p = Process(target=f, args=(q,))
    p2=Process(target=f2,args=(q,))
    p3=Process(target=f3,args=(q,))
    p.start()
    p2.start()
    p3.start()
    p.join()
    p2.join()
    p3.join()