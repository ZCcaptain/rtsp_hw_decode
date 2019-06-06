#include <stdio.h>
#include <stdlib.h>
#include "decode.h"

#include"nvJPEG.h"
#include"mat_warper.h"
#include <opencv2/opencv.hpp>
#include<pybind11/pybind11.h>
#include <core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/stitching.hpp>
#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <fstream>


namespace py = pybind11;
using namespace std;

int cpu_callback(const int flag, cv::Mat &image) {
	return 0;
}

py::array_t<unsigned char> jpeg_to_numpy(const char * img, int size){
	return  cv_mat_uint8_3c_to_numpy(decode_jpeg((const unsigned char *)(img), size));
}



void init_jpegDecoderAndEncoder(int gpu_id){
	findCudaDevice_new(gpu_id);
}

int gpu_callback(const int flag, char *image, int size) {
	py::bytes img = py::bytes(image, size);
	py::module sys = py::module::import("sys");
	py::module t = py::module::import("globalQueue");
	t.attr("q_put")(flag,img);
	return 0;
}

void decode_thread(string  rtsp_addr,int camera_id , string gpu_id, int quality, int interval)
{
	findCudaDevice_new(stoi(gpu_id));
	ffmpeg_video_decode(rtsp_addr,  camera_id,  cpu_callback,  gpu_callback, gpu_id ,quality, interval, true,  false );
}



int decode_to_jpeg(string  rtsp_addr, int camera_id, string gpu_id,int quality, int interval) {
	//ffmpeg初始化
	ffmpeg_global_init();
//	rtsp_addrs.push_back("rtsp://admin:admin12345@192.168.0.51:554/Streaming/Channels/101");
//	display_queue.push_back(new threadsafe_queue());
//	rtsp_addrs.push_back("rtsp://admin:admin12345@192.168.0.51:554/Streaming/Channels/101");
//	display_queue.push_back(new threadsafe_queue());
//	rtsp_addrs.push_back("rtsp://admin:admin12345@192.168.0.51:554/Streaming/Channels/101");
//	display_queue.push_back(new threadsafe_queue());
//
//
//	rtsp_addrs.push_back( "rtsp://admin:admin12345@1=92.168.0.141:554/Streaming/Channels/101");
//	display_queue.push_back(new threadsafe_queue());
//	rtsp_addrs.push_back( "rtsp://admin:admin12345@192.168.0.141:554/Streaming/Channels/101");
//	display_queue.push_back(new threadsafe_queue());
//	rtsp_addrs.push_back( "rtsp://admin:admin12345@192.168.0.141:554/Streaming/Channels/101");
//	display_queue.push_back(new threadsafe_queue());
//
//	rtsp_addrs.push_back( "rtsp://admin:admin12345@192.168.0.52:554/Streaming/Channels/101");
//	display_queue.push_back(new threadsafe_queue());
//	rtsp_addrs.push_back( "rtsp://admin:admin12345@192.168.0.52:554/Streaming/Channels/101");
//	display_queue.push_back(new threadsafe_queue());
//	rtsp_addrs.push_back( "rtsp://admin:admin12345@192.168.0.52:554/Streaming/Channels/101");
//	display_queue.push_back(new threadsafe_queue());
//	rtsp_addrs.push_back( "rtsp://admin:admin12345@192.168.0.52:554/Streaming/Channels/101");
//	display_queue.push_back(new threadsafe_queue());
//
//	rtsp_addrs.push_back( "rtsp://admin:sdkj123456@192.168.0.64:554/Streaming/Channels/101");
//	display_queue.push_back(new threadsafe_queue());
//	rtsp_addrs.push_back( "rtsp://admin:sdkj123456@192.168.0.64:554/Streaming/Channels/101");
//	display_queue.push_back(new threadsafe_queue());
//	rtsp_addrs.push_back( "rtsp://admin:sdkj123456@192.168.0.64:554/Streaming/Channels/101");
//	display_queue.push_back(new threadsafe_queue());
//	rtsp_addrs.push_back( "rtsp://admin:sdkj123456@192.168.0.64:554/Streaming/Channels/101");
//	display_queue.push_back(new threadsafe_queue());
	thread decode = thread(decode_thread,rtsp_addr, camera_id, gpu_id,quality, interval);
	decode.join();
	return 0;
}

PYBIND11_MODULE(decode, m) {
    m.def("decode_to_jpeg", &decode_to_jpeg);
    m.def("init_jpegDecoderAndEncoder", &init_jpegDecoderAndEncoder);
    m.def("jpeg_to_numpy", &jpeg_to_numpy);
}



