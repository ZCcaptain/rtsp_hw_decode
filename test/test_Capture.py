import decode
import cv2
from multiprocessing import Process 
from threading import Thread
import  globalQueue


def decode_start(rtsp_addr,cam_id, gpu_id,quality,interval):#add jpeg quality (1-100)
    #decode.init_jpegDecoderAndEncoder(1)
    decode.decode_to_jpeg(rtsp_addr,cam_id,gpu_id,quality,interval)


def display_thread(index):
        decode.init_jpegDecoderAndEncoder(0)#add function for init jpegDecoder
        count = 0
        #cv2.namedWindow(str(index)) #can't multi thread
        while True:
                if(count % 5 == 0):    #just test       
                        jpg = globalQueue.q_get(index)
                        #print("type(jpg):",type(jpg))
                        print(index, ': ',globalQueue.q_size(index)) 
                        #print(len(jpg))
                        img = decode.jpeg_to_numpy(jpg, len(jpg)) #add function jpeg_to_numpy 
                        #cv2.imshow(str(index),img)
                        #cv2.waitKey(1)
                else:
                        globalQueue.q_get(index)
                        #print(index, ': ',globalQueue.q_size(index)) 
                count += 1
                if(count == 10000):
                        count = 0


def main_thread(rtsp_addr,cam_id, gpu_id,quality,interval):
        td1 = Thread(target=decode_start,args=(rtsp_addr, cam_id, gpu_id,quality,interval))
        td2 = Thread(target=display_thread,args=(cam_id,))
        td1.start()
        td2.start()
        td1.join()
        td2.join()

globalQueue.initQueue()

rtsp_addrs = [] # rtsp address and queue
#rtsp_addrs.append("4.avi")
#globalQueue.addQueue()
rtsp_addrs.append("rtsp://admin:admin12345@192.168.0.51:554/Streaming/Channels/101")
globalQueue.addQueue()
rtsp_addrs.append("rtsp://admin:admin12345@192.168.0.51:554/Streaming/Channels/101")
globalQueue.addQueue()
rtsp_addrs.append("rtsp://admin:admin12345@192.168.0.51:554/Streaming/Channels/101")
globalQueue.addQueue()


#rtsp_addrs.append("5.avi")
#globalQueue.addQueue()
rtsp_addrs.append("rtsp://admin:admin12345@192.168.0.52:554/Streaming/Channels/101")
globalQueue.addQueue()
rtsp_addrs.append("rtsp://admin:admin12345@192.168.0.52:554/Streaming/Channels/101")
globalQueue.addQueue()
rtsp_addrs.append("rtsp://admin:admin12345@192.168.0.52:554/Streaming/Channels/101")
globalQueue.addQueue()


rtsp_addrs.append("rtsp://admin:admin12345@192.168.0.141:554/Streaming/Channels/101") 
globalQueue.addQueue()
rtsp_addrs.append("rtsp://admin:admin12345@192.168.0.141:554/Streaming/Channels/101")
globalQueue.addQueue()
rtsp_addrs.append("rtsp://admin:admin12345@192.168.0.141:554/Streaming/Channels/101")
globalQueue.addQueue()


rtsp_addrs.append("rtsp://admin:sdkj123456@192.168.0.64:554/Streaming/Channels/101")
globalQueue.addQueue()
rtsp_addrs.append("rtsp://admin:sdkj123456@192.168.0.64:554/Streaming/Channels/101")
globalQueue.addQueue()
rtsp_addrs.append("rtsp://admin:sdkj123456@192.168.0.64:554/Streaming/Channels/101")
globalQueue.addQueue()
rtsp_addrs.append("rtsp://admin:sdkj123456@192.168.0.64:554/Streaming/Channels/101")
globalQueue.addQueue()




decode_threads = []


for i in range(len(rtsp_addrs)):
        decode_threads.append(Process(target=main_thread,args=(rtsp_addrs[i], i, '0',100,1)))



for i in decode_threads:
        i.start()

for i in decode_threads:
        i.join() 


for i in decode_threads:
        i.terminate()

