/*
 * nvJPEG.h
 *
 *  Created on: May 8, 2019
 *      Author: sdkj
 */

#ifndef NVJPEG_H_
#define NVJPEG_H_
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "nvjpeg.h"
#include "helper_cuda.h"
#include "helper_timer.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <string.h>    // strcmpi
#include <sys/time.h>  // timings

#include <dirent.h>  // linux dir traverse
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

struct decode_params_t {
  std::string input_dir;
  int batch_size;
  int total_images;
  int dev;
  int warmup;

  nvjpegJpegState_t nvjpeg_state;
  nvjpegHandle_t nvjpeg_handle;
  cudaStream_t stream;

  nvjpegOutputFormat_t fmt;
  bool write_decoded;
  std::string output_dir;

  bool pipelined;
  bool batched;
};


int init_jpeg_decode();

void release_buffer(nvjpegImage_t &ibuf);
cv::Mat decode_jpeg(const unsigned char* image, int size);


inline int findCudaDevice_new(int gpu_id) {
  cudaDeviceProp deviceProp;
  int devID = 0;

  // If the command-line has a device number specified, use it
  if (gpu_id > 0) {
    devID = gpu_id;

    if (devID < 0) {
      printf("Invalid command line parameter\n ");
      exit(EXIT_FAILURE);
    } else {
      devID = gpuDeviceInit(devID);

      if (devID < 0) {
        printf("exiting...\n");
        exit(EXIT_FAILURE);
      }
    }
  } else {
    // Otherwise pick the device with highest Gflops/s
    devID = gpuGetMaxGflopsDeviceId();
    checkCudaErrors(cudaSetDevice(devID));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID,
           deviceProp.name, deviceProp.major, deviceProp.minor);
  }

  return devID;
}
#endif /* NVJPEG_H_ */
