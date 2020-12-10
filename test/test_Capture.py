import decode
import cv2
from multiprocessing import Process 
from threading import Thread
import  globalQueue
import datetime
import time
import numpy as np




def decode_start(rtsp_addr,cam_id, gpu_id,quality,interval):#add jpeg quality (1-100)
        decode.decode_to_jpeg(rtsp_addr,cam_id,gpu_id,quality,interval)
        



def decode_start_cpu(rtsp_addr,cam_id):
        cap = cv2.VideoCapture(rtsp_addr)
        # i = 0
        print("Camera Thread %s has been started" % cam_id)
        while True:
                if not cap.isOpened():
                        cap.release()
                        time.sleep(10)
                        print('摄像头%s断开，正在重连。' % (cam_id))
                        cap = cv2.VideoCapture(rtsp_addr)
                res, image = cap.read()
                if res == False:
                        cap.release()
                        cap = cv2.VideoCapture(rtsp_addr)
                        print("Resource False.")
                        continue
                
                globalQueue.q_put(cam_id, image)



def display_thread(index):
        decode.init_jpegDecoderAndEncoder(0)#add function for init jpegDecoder
        count = 0
        #cv2.namedWindow(str(index)) #can't multi thread
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

        while True:
                if(count % 10 == 0):    #just test       
                        
                        img = globalQueue.q_get(index)
                        # print("type(jpg):",type(jpg))
                        print(index, ': ',globalQueue.q_size(index)) 
                        # print(img.shape)
                        # jpg = decode.imencode(img, img.shape[1] , img.shape[0])
                        # print(len(jpg))
                        # cv2.imwrite('./1.jpg', img)
                        # with open('./' + str(count) + '.jpg', 'wb') as f:
                                # f.write(jpg)
                        # result, encimg = cv2.imencode('.jpg', img, encode_param)
                        # print(len(jpg))
                        # img = decode.jpeg_to_numpy(jpg, len(jpg)) #add function jpeg_to_numpy 
                        #cv2.imshow(str(index),img)
                        #cv2.waitKey(1)
                else:
                        globalQueue.q_get(index)
                        #print(index, ': ',globalQueue.q_size(index)) 
                count += 1
                if(count == 10000):
                        count = 0
def display_thread_cpu(index):
        count = 0
        # cv2.namedWindow(str(index)) #can't multi thread
        while True:
                if(count % 10 == 0):    #just test       
                        img = globalQueue.q_get(index)
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                        result, encimg = cv2.imencode('.jpg', img, encode_param)
                        print(index, ': ',globalQueue.q_size(index)) 

                else:
                        globalQueue.q_get(index)
                count += 1
                if(count == 10000):
                        count = 0


def main_thread(rtsp_addr,cam_id, gpu_id,quality,interval):
        td1 = Thread(target=decode_start,args=(rtsp_addr, cam_id, gpu_id,quality,interval))
        td2 = Thread(target=display_thread,args=(cam_id,))
        # td1 = Thread(target=decode_start_cpu,args=(rtsp_addr, cam_id))
        # td2 = Thread(target=display_thread_cpu,args=(cam_id,))
        td1.start()
        td2.start()
        td1.join()
        td2.join()









globalQueue.initQueue()

rtsp_addrs = [] # rtsp address and queue
#rtsp_addrs.append("4.avi")
# #globalQueue.addQueue()
rtsp_addrs.append("rtsp://admin:admin12345@192.168.0.51:554/Streaming/Channels/101")
globalQueue.addQueue()
# rtsp_addrs.append("rtsp://admin:admin12345@192.168.0.51:554/Streaming/Channels/101")
# globalQueue.addQueue()
# rtsp_addrs.append("rtsp://admin:admin12345@192.168.0.51:554/Streaming/Channels/101")
# globalQueue.addQueue()



# #rtsp_addrs.append("5.avi")
# #globalQueue.addQueue()
rtsp_addrs.append("rtsp://admin:admin12345@192.168.0.52:554/Streaming/Channels/101")
globalQueue.addQueue()
# rtsp_addrs.append("rtsp://admin:admin12345@192.168.0.52:554/Streaming/Channels/101")
# globalQueue.addQueue()
# rtsp_addrs.append("rtsp://admin:admin12345@192.168.0.52:554/Streaming/Channels/101")
# globalQueue.addQueue()


rtsp_addrs.append("rtsp://admin:admin12345@192.168.0.141:554/Streaming/Channels/101") 
globalQueue.addQueue()
# rtsp_addrs.append("rtsp://admin:admin12345@192.168.0.141:554/Streaming/Channels/101")
# globalQueue.addQueue()
# rtsp_addrs.append("rtsp://admin:admin12345@192.168.0.141:554/Streaming/Channels/101")
# globalQueue.addQueue()


rtsp_addrs.append("rtsp://admin:sdkj123456@192.168.0.64:554/Streaming/Channels/101")
globalQueue.addQueue()
# rtsp_addrs.append("rtsp://admin:sdkj123456@192.168.0.64:554/Streaming/Channels/101")
# globalQueue.addQueue()
# rtsp_addrs.append("rtsp://admin:sdkj123456@192.168.0.64:554/Streaming/Channels/101")
# globalQueue.addQueue()
# rtsp_addrs.append("rtsp://admin:sdkj123456@192.168.0.64:554/Streaming/Channels/101")
# globalQueue.addQueue()




decode_threads = []


for i in range(len(rtsp_addrs)):
        decode_threads.append(Process(target=main_thread,args=(rtsp_addrs[i], i, '0',100,1)))



for i in decode_threads:
        i.start()

for i in decode_threads:
        i.join() 


for i in decode_threads:
        i.terminate()



