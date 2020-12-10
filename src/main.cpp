/**  
 *  Copyright (c) 2019 All Rights Reserved
 *  @author Zhao Chao
 *  @date 2019.06.06 15:01:21
 *  @brief decode
 */
#include <stdio.h>
#include <stdlib.h>
#include "decode.h"

#include"nvJPEG.h"
#include"mat_warper.h"
#include "threadsafe_queue.h"
#include <opencv2/opencv.hpp>
#include<pybind11/pybind11.h>
#include <core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/cudacodec.hpp>
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

py::bytes imencode(py::array_t<unsigned char>& input, int height, int width){
	cv::Mat img  = numpy_uint8_3c_to_cv_mat(input);
	return jpeg_encode(img, height, width);
}

void init_jpegDecoderAndEncoder(int gpu_id){
	findCudaDevice_new(gpu_id);
}


int gpu_callback(const int flag, cv::cuda::GpuMat &image) {
	cv::Mat mat(image.cols, image.rows, CV_8UC3);
	image.download(mat);
    py::module t = py::module::import("globalQueue");
	t.attr("q_put")(flag, cv_mat_uint8_3c_to_numpy(mat));
	return 0;
}


// int gpu_callback(const int flag, char *image, int size) {
// 	py::bytes img = py::bytes(image, size);
// 	py::module sys = py::module::import("sys");
// 	py::module t = py::module::import("globalQueue");
// 	t.attr("q_put")(flag,img);
// 	return 0;
// }

void decode_thread(string  rtsp_addr,int camera_id , string gpu_id, int quality, int interval)
{
	ffmpeg_video_decode(rtsp_addr,  camera_id,  cpu_callback,  gpu_callback, gpu_id ,quality, interval, true,  false );
}


// py::array_t<unsigned char> s_read(){

// 			cv::Mat temp;
// 			while(s->empty());
// 			s->wait_and_pop(temp);
// 			return cv_mat_uint8_3c_to_numpy(temp);
			
// }

// int s_size(){
// 	return s->size();
// }


int decode_to_jpeg(string  rtsp_addr, int camera_id, string gpu_id,int quality, int interval) {
	//ffmpeg初始化
	ffmpeg_global_init();
	thread decode = thread(decode_thread,rtsp_addr, camera_id, gpu_id,quality, interval);
	decode.join();
	return 0;
}


// int testOpencvDecode(string rtsp_addr){
// 	cv::cuda::GpuMat d_frame;
//     cv::Ptr<cv::cudacodec::VideoReader> d_reader = cv::cudacodec::createVideoReader(rtsp_addr);
// 	int count = 0;
// 	for (;;)
// 	{
// 		if (!d_reader->nextFrame(d_frame))  //BRGA格式
// 			break;
// 		//   gpu_frame_count++;
// 		cv::Mat frame2;
// 		d_frame.download(frame2);
// 		cv::imwrite("xxx.png", frame2);
// 	}
// 	return 0;
// }

// 测试线程类
// class TestThread{
// 	private:
// 		threadsafe_queue *q;
// 		thread decode;
// 		int opened;
// 		string addr;
// 	public:
// 		TestThread(const std::string & addr, int camera_id, string gpu_id,int quality, int interval):decode(std::bind(&TestThread::decode_test,this)){
// 			this->q = new threadsafe_queue();
// 			// decode = thread(decode_test, 1);
// 			opened = 0;
// 			thread decode = thread(decode_thread,addr, camera_id, gpu_id,quality, interval);
// 			decode.detach();
// 		}
// 		 void decode_test(){
// 			// opened = 0;
// 			while(1){
// 				cv::Mat temp = cv::Mat::zeros(3,3,CV_8UC3);
// 				this->q->push(temp);
// 				// opened = 1;
// 			}
// 		}

// 		int is_opened(){return this->opened;} 

// 		py::array_t<unsigned char> read(){

// 			cv::Mat temp;
// 			if(this->q->try_pop(temp))

// 				return cv_mat_uint8_3c_to_numpy(temp);
// 			else
// 			{
// 				return cv_mat_uint8_3c_to_numpy(temp);
// 			}
			
// 		}
// };









PYBIND11_MODULE(decode, m) {
    m.def("decode_to_jpeg", &decode_to_jpeg);
    m.def("init_jpegDecoderAndEncoder", &init_jpegDecoderAndEncoder);
    m.def("jpeg_to_numpy", &jpeg_to_numpy);
	m.def("imencode", &imencode);
	
}



