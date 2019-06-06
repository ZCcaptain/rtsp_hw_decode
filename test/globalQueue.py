from queue import Queue


def initQueue():
    global q
    q = []
    
def addQueue():
    tmp = Queue()
    q.append(tmp)

def q_put(index,jpeg_bytes):
    if(q[index].qsize() < 500):
        q[index].put(jpeg_bytes)
    #if(q[index].qsize() > 1):
        #q[index].get() 
        #print("lost mat")
    else:
        print(index, "lost jpeg_bytes")


def q_get(index):
    return q[index].get(True)
    

def q_size(index):
    return q[index].qsize()

def q_count():
    return len(q)





