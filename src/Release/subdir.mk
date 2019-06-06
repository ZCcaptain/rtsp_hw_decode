################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../decode.cpp \
../gh_jpegnpp.cpp \
../main.cpp \
../mat_warper.cpp \
../nvJPEG.cpp 

OBJS += \
./decode.o \
./gh_jpegnpp.o \
./main.o \
./mat_warper.o \
./nvJPEG.o 

CPP_DEPS += \
./decode.d \
./gh_jpegnpp.d \
./main.d \
./mat_warper.d \
./nvJPEG.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.1/bin/nvcc -I/home/sdkj/software/opencv-3.4.1/include -I/usr/local/include/pybind11 -I/root/anaconda3/include/python3.7m -I/home/sdkj/software/opencv-3.4.1/include/opencv -I/home/sdkj/software/opencv-3.4.1/include/opencv2 -I/home/sdkj/software/ffmpeg-4.1/include -I/usr/local/cuda-10.1/samples/common/inc -I/usr/local/cuda-10.1/samples/7_CUDALibraries/common/UtilNPP -O3 -Xcompiler -fPIC -std=c++11 -gencode arch=compute_75,code=sm_75  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	 /usr/local/cuda-10.1/bin/nvcc -I/home/sdkj/software/opencv-3.4.1/include -I/usr/local/include/pybind11 -I/root/anaconda3/include/python3.7m -I/home/sdkj/software/opencv-3.4.1/include/opencv -I/home/sdkj/software/opencv-3.4.1/include/opencv2 -I/home/sdkj/software/ffmpeg-4.1/include -I/usr/local/cuda-10.1/samples/common/inc -I/usr/local/cuda-10.1/samples/7_CUDALibraries/common/UtilNPP -O3 -Xcompiler -fPIC -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


