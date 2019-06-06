#ifndef __JPEG_NPP_H__
#define __JPEG_NPP_H__

#define IMAGE_MAX_WIDTH  1920
#define IMAGE_MAX_HEIGHT 1080

//每个线程使用一个静态实例。
#define USE_STATIC_INSTANCE	1

//typedef可以严格限定值的范围。
typedef enum
{
    IMAGE_DEST_FILE,
    IMAGE_DEST_MEM
} IMAGE_DEST;

typedef enum
{
    MEM_HOST,
    MEM_CUDA
} MEM_TYPE;

typedef struct
{
    int width;
    int height;
} RectSize;

//缓冲区的通用定义。
typedef struct
{
    //数据存储区。如果发现容量不够，需要先free再malloc
	uint8_t* data;
    //数据缓冲区的容量
    int   capacity;
    //数据实际长度，memcpy。<=capacity
    int   size;

    MEM_TYPE mem_type;
    //如果多个线程使用，需要使用这个来决定释放
    //int ref_count;
} DataBuffer;


int jpeg_npp_dest(const IMAGE_DEST dest, char *pDest, const int quality,
        const DataBuffer* pImageBuffer, const RectSize* pImageSize, const int channel);

int jpeg_npp_file(     const char *szOutputFile,      const int quality, 
        const DataBuffer* pImageBuffer, const RectSize* pImageSize, const int channel);

int jpeg_npp_mem(      char  *pJpegBuffer, const int quality,
        const DataBuffer* pImageBuffer, const RectSize* pImageSize, const int channel);



#endif //__JPEG_NPP_H__
