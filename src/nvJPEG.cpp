/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// This sample needs at least CUDA 10.0. It demonstrates usages of the nvJPEG
// library nvJPEG supports single and multiple image(batched) decode. Multiple
// images can be decoded using the API for batch mode

#include"nvJPEG.h"
int dev_malloc(void **p, size_t s) { return (int)cudaMalloc(p, s); }

int dev_free(void *p) { return (int)cudaFree(p); }

typedef std::vector<std::string> FileNames;
typedef std::vector<std::vector<char> > FileData;


int num = 0;
decode_params_t params;
cudaDeviceProp props;


int writeBMP(const char *filename, const unsigned char *d_chanR, int pitchR,
             const unsigned char *d_chanG, int pitchG,
             const unsigned char *d_chanB, int pitchB, int width, int height) {
  unsigned int headers[13];
  FILE *outfile;
  int extrabytes;
  int paddedsize;
  int x;
  int y;
  int n;
  int red, green, blue;

  std::vector<unsigned char> vchanR(height * width);
  std::vector<unsigned char> vchanG(height * width);
  std::vector<unsigned char> vchanB(height * width);
  unsigned char *chanR = vchanR.data();
  unsigned char *chanG = vchanG.data();
  unsigned char *chanB = vchanB.data();
  checkCudaErrors(cudaMemcpy2D(chanR, (size_t)width, d_chanR, (size_t)pitchR,
                               width, height, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy2D(chanG, (size_t)width, d_chanG, (size_t)pitchR,
                               width, height, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy2D(chanB, (size_t)width, d_chanB, (size_t)pitchR,
                               width, height, cudaMemcpyDeviceToHost));

  extrabytes =
      4 - ((width * 3) % 4);  // How many bytes of padding to add to each
  // horizontal line - the size of which must
  // be a multiple of 4 bytes.
  if (extrabytes == 4) extrabytes = 0;

  paddedsize = ((width * 3) + extrabytes) * height;

  // Headers...
  // Note that the "BM" identifier in bytes 0 and 1 is NOT included in these
  // "headers".

  headers[0] = paddedsize + 54;  // bfSize (whole file size)
  headers[1] = 0;                // bfReserved (both)
  headers[2] = 54;               // bfOffbits
  headers[3] = 40;               // biSize
  headers[4] = width;            // biWidth
  headers[5] = height;           // biHeight

  // Would have biPlanes and biBitCount in position 6, but they're shorts.
  // It's easier to write them out separately (see below) than pretend
  // they're a single int, especially with endian issues...

  headers[7] = 0;           // biCompression
  headers[8] = paddedsize;  // biSizeImage
  headers[9] = 0;           // biXPelsPerMeter
  headers[10] = 0;          // biYPelsPerMeter
  headers[11] = 0;          // biClrUsed
  headers[12] = 0;          // biClrImportant

  if (!(outfile = fopen(filename, "wb"))) {
    std::cerr << "Cannot open file: " << filename << std::endl;
    return 1;
  }

  //
  // Headers begin...
  // When printing ints and shorts, we write out 1 character at a time to avoid
  // endian issues.
  //
  fprintf(outfile, "BM");

  for (n = 0; n <= 5; n++) {
    fprintf(outfile, "%c", headers[n] & 0x000000FF);
    fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
    fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
    fprintf(outfile, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
  }

  // These next 4 characters are for the biPlanes and biBitCount fields.

  fprintf(outfile, "%c", 1);
  fprintf(outfile, "%c", 0);
  fprintf(outfile, "%c", 24);
  fprintf(outfile, "%c", 0);

  for (n = 7; n <= 12; n++) {
    fprintf(outfile, "%c", headers[n] & 0x000000FF);
    fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
    fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
    fprintf(outfile, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
  }

  //
  // Headers done, now write the data...
  //

  for (y = height - 1; y >= 0;
       y--)  // BMP image format is written from bottom to top...
  {
    for (x = 0; x <= width - 1; x++) {
      red = chanR[y * width + x];
      green = chanG[y * width + x];
      blue = chanB[y * width + x];

      if (red > 255) red = 255;
      if (red < 0) red = 0;
      if (green > 255) green = 255;
      if (green < 0) green = 0;
      if (blue > 255) blue = 255;
      if (blue < 0) blue = 0;
      // Also, it's written in (b,g,r) format...

      fprintf(outfile, "%c", blue);
      fprintf(outfile, "%c", green);
      fprintf(outfile, "%c", red);
    }
    if (extrabytes)  // See above - BMP lines must be of lengths divisible by 4.
    {
      for (n = 1; n <= extrabytes; n++) {
        fprintf(outfile, "%c", 0);
      }
    }
  }

  fclose(outfile);
  return 0;
}

uchar *img = new uchar[1920*1080 * 3];
cv::Mat writeMat(const unsigned char *d_chanR, int pitchR,
             const unsigned char *d_chanG, int pitchG,
             const unsigned char *d_chanB, int pitchB, int width, int height) {
  int x;
  int y;
  int n = 0;
  uchar red, green, blue;

  std::vector<unsigned char> vchanR(height * width);
  std::vector<unsigned char> vchanG(height * width);
  std::vector<unsigned char> vchanB(height * width);
  unsigned char *chanR = vchanR.data();
  unsigned char *chanG = vchanG.data();
  unsigned char *chanB = vchanB.data();
  checkCudaErrors(cudaMemcpy2D(chanR, (size_t)width, d_chanR, (size_t)pitchR,
                               width, height, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy2D(chanG, (size_t)width, d_chanG, (size_t)pitchR,
                               width, height, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy2D(chanB, (size_t)width, d_chanB, (size_t)pitchR,
                               width, height, cudaMemcpyDeviceToHost));


  for (y = 0; y<height;
       y++)
  {
    for (x = 0; x < width; x++) {
      red = chanR[y * width + x];
      green = chanG[y * width + x];
      blue = chanB[y * width + x];

      if (red > 255) red = 255;
      if (red < 0) red = 0;
      if (green > 255) green = 255;
      if (green < 0) green = 0;
      if (blue > 255) blue = 255;
      if (blue < 0) blue = 0;
      // Also, it's written in (b,g,r) format...
      img[n++] = blue;
      img[n++] = green;
      img[n++] = red;

    }
  }
  return  cv::Mat(height, width, CV_8UC3, img);
}






int init_jpeg_decode() {


  params.input_dir = "./";

  params.batch_size = 1;

  params.total_images = 1;

//  params.dev = gpu_id;


//std::cout << "......."<< std::endl;
  params.warmup = 0;

  params.batched = false;

  params.pipelined = false;

  params.fmt = NVJPEG_OUTPUT_BGR;

  params.write_decoded = false;



//  checkCudaErrors(cudaGetDeviceProperties(&props, params.dev));
//  printf("Using GPU %d (%s, %d SMs, %d th/SM max, CC %d.%d, ECC %s)\n",
//           params.dev, props.name, props.multiProcessorCount,
//           props.maxThreadsPerMultiProcessor, props.major, props.minor,
//           props.ECCEnabled ? "on" : "off");

  nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
  checkCudaErrors(nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &dev_allocator,
                               &params.nvjpeg_handle));
  checkCudaErrors(
      nvjpegJpegStateCreate(params.nvjpeg_handle, &params.nvjpeg_state));
  checkCudaErrors(
      nvjpegDecodeBatchedInitialize(params.nvjpeg_handle, params.nvjpeg_state,
                                    params.batch_size, 1, params.fmt));

  // read source images
  //FileNames image_names;
  //readInput(params.input_dir, image_names);

  if (params.total_images == -1) {
  //  params.total_images = image_names.size();
  } else if (params.total_images % params.batch_size) {
    params.total_images =
        ((params.total_images) / params.batch_size) * params.batch_size;
  }





  //checkCudaErrors(nvjpegJpegStateDestroy(params.nvjpeg_state));
  //checkCudaErrors(nvjpegDestroy(params.nvjpeg_handle));

  return EXIT_SUCCESS;
}

void release_buffer(nvjpegImage_t &ibuf) {

	for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++)
	  if (ibuf.channel[c]) checkCudaErrors(cudaFree(ibuf.channel[c]));

}

cv::Mat  decode_jpeg(const unsigned char* image, int size){
	init_jpeg_decode();

	int widths[NVJPEG_MAX_COMPONENT];
	int heights[NVJPEG_MAX_COMPONENT];
	int channels;
	nvjpegChromaSubsampling_t subsampling;
	nvjpegImage_t iout;
	nvjpegImage_t isz;
	checkCudaErrors(
	      cudaStreamCreateWithFlags(&params.stream, cudaStreamNonBlocking));
	//init iout
	for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
	  iout.channel[c] = NULL;
	  iout.pitch[c] = 0;
	  isz.pitch[c] = 0;
	 }

	checkCudaErrors(nvjpegGetImageInfo(
		params.nvjpeg_handle, (unsigned char *)image, size,
		&channels, &subsampling, widths, heights));
	int mul = 1;
	if(params.fmt == NVJPEG_OUTPUT_BGR){
		channels = 3;
		widths[1] = widths[2] = widths[0];
		heights[1] = heights[2] = heights[0];
	}
	for (int c = 0; c < channels; c++) {
	      int aw = mul * widths[c];
	      int ah = heights[c];
	      int sz = aw * ah;
	      iout.pitch[c] = aw;
	      if (sz > isz.pitch[c]) {
	        if (iout.channel[c]) {
	          checkCudaErrors(cudaFree(iout.channel[c]));
	        }
	        checkCudaErrors(cudaMalloc(&iout.channel[c], sz));
	        isz.pitch[c] = sz;
	      }
	}
	checkCudaErrors(cudaStreamSynchronize(params.stream));
	int thread_idx = 0;
    for (int i = 0; i < params.batch_size; i++) {
      checkCudaErrors(nvjpegDecode(params.nvjpeg_handle, params.nvjpeg_state,
                                   (const unsigned char *)image,
                                   size, params.fmt, &iout,
                                   params.stream));
	checkCudaErrors(cudaStreamSynchronize(params.stream));
    }
   // std::cout << "1" << std ::endl;

	//mat.create(heights[0], widths[0], CV_8UC3);
	//uchar *img = new uchar[heights[0] * widths[0] * 3];
	//int idx;
	//int t;
	 //std::cout << "1" << std ::endl;
	//for(int i = 0; i < heights[0]; i++){
	//	for(int j = 0; j < widths[0]; j++){
	//		for (int k = 0; k < 3; k++) {
	//			t = i * widths[0] + j;
	//			idx = (i * widths[0] + j) * 3 + k;
	//			std::cout << t << " " << idx << std::endl;
				//if ((uint8_t)(iout.channel[k][t]) >=0 && (uint8_t)(iout.channel[k][t]) < 255) {
				//	std::cout << iout.channel[k][t] << std::endl;
				//	img[idx] = iout.channel[k][t];
				//} else {
				//	std::cout << iout.channel[k][t] << std::endl;
				//	img[idx] = iout.channel[k][t] < 0 ? 0 : 255;
				//}
		//	}
	//	}
	//}
	// std::cout << "mat" << std ::endl;
	//cv::Mat mat = cv::Mat(heights[0], widths[0], CV_8UC3, img);



	/*char filename[100] = {0};
	snprintf(filename, 100, "./pic/out_%d.bmp", num ++);


	writeBMP(filename, iout.channel[2], iout.pitch[2],
	                     iout.channel[1], iout.pitch[1], iout.channel[0],
	                     iout.pitch[0], widths[0], heights[0]);*/

    cv::Mat mat = writeMat(iout.channel[2], iout.pitch[2],
    	                     iout.channel[1], iout.pitch[1], iout.channel[0],
    	                     iout.pitch[0], widths[0], heights[0]);
	release_buffer(iout);
	checkCudaErrors(cudaStreamDestroy(params.stream));

	checkCudaErrors(nvjpegJpegStateDestroy(params.nvjpeg_state));
	checkCudaErrors(nvjpegDestroy(params.nvjpeg_handle));
	return mat;
}









