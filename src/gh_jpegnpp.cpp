
#include <npp.h>
#include <cuda_runtime.h>
#include <Exceptions.h>

#include "Endianess.h"
#include <math.h>
#include <cmath>
#include <string.h>
#include <fstream>
#include <iostream>

#include <helper_string.h>
#include <helper_cuda.h>

#include "gh_jpegnpp.h"


//量化表数据
typedef struct 
{
    unsigned char nPrecisionAndIdentifier;
    unsigned char aTable[64];
} QuantizationTable;

//图片信息
typedef struct 
{
    unsigned char  nSamplePrecision;
    unsigned short nHeight;
    unsigned short nWidth;
    unsigned char  nComponents;
    unsigned char  aComponentIdentifier[3];
    unsigned char  aSamplingFactors[3];
    unsigned char  aQuantizationTableSelector[3];
} FrameHeader;

//扫描头
typedef struct 
{
    unsigned char nComponents;
    unsigned char aComponentSelector[3];
    unsigned char aHuffmanTablesSelector[3];
    unsigned char nSs;
    unsigned char nSe;
    unsigned char nA;
} ScanHeader;

//霍夫曼编码表数据，一般是固定的
typedef struct
{
    unsigned char nClassAndIdentifier;
    unsigned char aCodes[16];
    unsigned char aTable[256];
} HuffmanTable;


class CudaJpegEncode
{
public:
    CudaJpegEncode();
    ~CudaJpegEncode();
public:
    void Init(int quality, const RectSize* pImageSize);
    void SetQuality(unsigned char* pTable, const unsigned char* pTable50, int quality);
    void Release();
public:
    int  EncodeJpeg(const IMAGE_DEST dest, char* pDest);
    void SetData(const DataBuffer* pImageBuffer, int yuv_fmt, const RectSize* pImageSize);
public:
    NppiDCTState*        pDCTState;
    FrameHeader          oFrameHeader;
    QuantizationTable    aQuantizationTables[4];
    Npp8u*               pdQuantizationTables;

    HuffmanTable         aHuffmanTables[4];
    //HuffmanTable*        pHuffmanDCTables = aHuffmanTables;
    //HuffmanTable*        pHuffmanACTables = &aHuffmanTables[2];
    HuffmanTable*        pHuffmanDCTables;
    HuffmanTable*        pHuffmanACTables;
    ScanHeader           oScanHeader;

    NppiSize aSrcSize[3];
    Npp16s  *apdDCT[3];
    Npp32s   aDCTStep[3];

    Npp8u   *apSrcImage[3];
    Npp32s   aSrcImageStep[3];
    size_t   aSrcPitch[3];

    NppiEncodeHuffmanSpec *apHuffmanDCTable[3];
    NppiEncodeHuffmanSpec *apHuffmanACTable[3];

    int nMCUBlocksH;
    int nMCUBlocksV;

    Npp8u *pdScan;
    Npp32s nScanSize;

    Npp8u *pJpegEncoderTemp;
    size_t nTempSize;

    unsigned char *pDstJpeg;
public:
    uint8_t*    mY;
    uint8_t*    mU;
    uint8_t*    mV;
    int         mWidth;
    int         mHeight;
};



#define  MEMORY_ALGN_DEVICE     511
#define  HD_MEMORY_ALGN_DEVICE  511

#define  YUV_FMT_NV12           1
#define  YUV_FMT_NV21           2
#define  YUV_FMT_YUV420         3

#ifndef max 
#define max(X, Y) ((X) > (Y) ?(X):(Y))
#endif 
//霍夫曼编码表
unsigned char STD_DC_Y_NRCODES[16] = { 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 };
unsigned char STD_DC_Y_VALUES[12] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

unsigned char STD_DC_UV_NRCODES[16] = { 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 };
unsigned char STD_DC_UV_VALUES[12] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

unsigned char STD_AC_Y_NRCODES[16] = { 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0X7D };
unsigned char STD_AC_Y_VALUES[162] =
{
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
    0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
    0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
    0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
    0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
    0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
    0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
    0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
    0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa
};

unsigned char STD_AC_UV_NRCODES[16] = { 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0X77 };
unsigned char STD_AC_UV_VALUES[162] =
{
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
    0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
    0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
    0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
    0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
    0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
    0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
    0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
    0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
    0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
    0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa
};

#define DCTSIZE2 64
#if 0
//量化表，可以根据图像质量要求更改，这里图像是90%
static const unsigned char std_Y_QT[DCTSIZE2] =
{
     3,  2,  2,  3,  2,  2,  3,  3,
     3,  3,  4,  3,  3,  4,  5,  8,
     5,  5,  4,  4,  5, 10,  7,  7,
     6,  8, 12, 10, 12, 12, 11, 10,
    11, 11, 13, 14, 18, 16, 13, 14,
    17, 14, 11, 11, 16, 22, 16, 17,
    19, 20, 21, 21, 21, 12, 15, 23,
    24, 22, 20, 24, 18, 20, 21, 20
};

static const unsigned char std_UV_QT[DCTSIZE2] =
{
     3,  4,  4,  5,  4,  5,  9,  5,
     5,  9, 20, 13, 11, 13, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20
};

#else
/* These are the sample quantization tables given in JPEG spec section K.1.
 * The spec says that the values given produce "good" quality, and
 * when divided by 2, "very good" quality.
 */
static const unsigned char std_Y_QT[DCTSIZE2] = {
  16,  11,  10,  16,  24,  40,  51,  61,
  12,  12,  14,  19,  26,  58,  60,  55,
  14,  13,  16,  24,  40,  57,  69,  56,
  14,  17,  22,  29,  51,  87,  80,  62,
  18,  22,  37,  56,  68, 109, 103,  77,
  24,  35,  55,  64,  81, 104, 113,  92,
  49,  64,  78,  87, 103, 121, 120, 101,
  72,  92,  95,  98, 112, 100, 103,  99
};
static const unsigned char std_UV_QT[DCTSIZE2] = {
  17,  18,  24,  47,  99,  99,  99,  99,
  18,  21,  26,  66,  99,  99,  99,  99,
  24,  26,  56,  99,  99,  99,  99,  99,
  47,  66,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99
};
#endif

int DivUp(int x, int d)
{
    return (x + d - 1) / d;
}

void writeMarker(unsigned char nMarker, unsigned char *&pData)
{
    *pData++ = 0x0ff;
    *pData++ = nMarker;
}


template<typename T>
void writeAndAdvance(unsigned char *&pData, T nElement)
{
    writeBigEndian<T>(pData, nElement);
    pData += sizeof(T);
}

void writeJFIFTag(unsigned char *&pData)
{
    const char JFIF_TAG[] =
    {
        0x4a, 0x46, 0x49, 0x46, 0x00,
        0x01, 0x02,
        0x00,
        0x00, 0x01, 0x00, 0x01,
        0x00, 0x00
    };

    writeMarker(0x0e0, pData);
    writeAndAdvance<unsigned short>(pData, sizeof(JFIF_TAG) + sizeof(unsigned short));
    memcpy(pData, JFIF_TAG, sizeof(JFIF_TAG));
    pData += sizeof(JFIF_TAG);
}

void writeQuantizationTable(const QuantizationTable &table, unsigned char *&pData)
{
    writeMarker(0x0DB, pData);
    writeAndAdvance<unsigned short>(pData, sizeof(QuantizationTable) + 2);
    memcpy(pData, &table, sizeof(QuantizationTable));
    pData += sizeof(QuantizationTable);
}


void writeHuffmanTable(const HuffmanTable &table, unsigned char *&pData)
{
    writeMarker(0x0C4, pData);

    // Number of Codes for Bit Lengths [1..16]
    int nCodeCount = 0;

    for (int i = 0; i < 16; ++i)
    {
        nCodeCount += table.aCodes[i];
    }

    writeAndAdvance<unsigned short>(pData, 17 + nCodeCount + 2);
    memcpy(pData, &table, 17 + nCodeCount);
    pData += 17 + nCodeCount;
}

void writeFrameHeader(const FrameHeader &header, unsigned char *&pData)
{
    unsigned char aTemp[128];
    unsigned char *pTemp = aTemp;

    writeAndAdvance<unsigned char >(pTemp, header.nSamplePrecision);
    writeAndAdvance<unsigned short>(pTemp, header.nHeight);
    writeAndAdvance<unsigned short>(pTemp, header.nWidth);
    writeAndAdvance<unsigned char >(pTemp, header.nComponents);

    for (int c = 0; c < header.nComponents; ++c)
    {
        writeAndAdvance<unsigned char>(pTemp, header.aComponentIdentifier[c]);
        writeAndAdvance<unsigned char>(pTemp, header.aSamplingFactors[c]);
        writeAndAdvance<unsigned char>(pTemp, header.aQuantizationTableSelector[c]);
    }

    unsigned short nLength = (unsigned short)(pTemp - aTemp);

    writeMarker(0x0C0, pData);
    writeAndAdvance<unsigned short>(pData, nLength + 2);
    memcpy(pData, aTemp, nLength);
    pData += nLength;
}

void writeScanHeader(const ScanHeader &header, unsigned char *&pData)
{
    unsigned char aTemp[128];
    unsigned char *pTemp = aTemp;

    writeAndAdvance<unsigned char>(pTemp, header.nComponents);

    for (int c = 0; c < header.nComponents; ++c)
    {
        writeAndAdvance<unsigned char>(pTemp, header.aComponentSelector[c]);
        writeAndAdvance<unsigned char>(pTemp, header.aHuffmanTablesSelector[c]);
    }

    writeAndAdvance<unsigned char>(pTemp, header.nSs);
    writeAndAdvance<unsigned char>(pTemp, header.nSe);
    writeAndAdvance<unsigned char>(pTemp, header.nA);

    unsigned short nLength = (unsigned short)(pTemp - aTemp);

    writeMarker(0x0DA, pData);
    writeAndAdvance<unsigned short>(pData, nLength + 2);
    memcpy(pData, aTemp, nLength);
    pData += nLength;
}


CudaJpegEncode::CudaJpegEncode()
{
    pDCTState   = NULL;

    nMCUBlocksH = 0;
    nMCUBlocksV = 0;

    pdScan      = NULL;
    nScanSize   = 0;

    pdQuantizationTables = NULL;
    pJpegEncoderTemp     = NULL;
    nTempSize            = 0;

    pDstJpeg = NULL;
    mY = NULL;
    mU = NULL;
    mV = NULL;

    this->pHuffmanDCTables = aHuffmanTables;
    this->pHuffmanACTables = &aHuffmanTables[2];

    this->nMCUBlocksH = 0;
    this->nMCUBlocksV = 0;
}


CudaJpegEncode::~CudaJpegEncode()
{
    Release();
}

/**
  初步测试，画质无变化。
  比如说，使用摄像头时：
  jpeglib产生的图片，左上的日期跟实际画面一样的明亮。
  而这边的日期，设置100画质，文件大小变化了，还是要暗一些的。
 */
void CudaJpegEncode::SetQuality(unsigned char* pTable, const unsigned char* pTable50, const int quality)
{
    int force_baseline = 1;
    int i=0;

    int scale_factor = quality;
    /* Safety limit on quality factor.  Convert 0 to 1 to avoid zero divide. */
    if (scale_factor <= 0)
    {
        scale_factor = 1;
    }
    else if (scale_factor > 100)
    {
        scale_factor = 100;
    }

    if (scale_factor < 50)
    {
        scale_factor = 5000 / scale_factor;
    }
    else
    {
        scale_factor = 200 - scale_factor*2;
    }

    for (i = 0; i < DCTSIZE2; i++)
    {
        //原来代码是long，并无必要。这里的50，是指默认表质量为50的意思
        int temp = ( pTable50[i] * scale_factor + 50) / 100;
        if (temp <= 0)
        {
            temp = 1;
        }
        else if (temp > 0x7FFF)
        {
            /* max quantizer needed for 12 bits */
            //32767不如使用0x8FFF
            temp = 0x7FFF;
        }

        if (force_baseline && temp > 0xFF)
        {
            //255L不如使用0xFF
            temp = 0xFF;
        }
        pTable[i] = (unsigned int) temp;
    }
}

/*
brief: init the jpeg encoder
*/
void CudaJpegEncode::Init(int quality, const RectSize* pImageSize)
{
#if USE_STATIC_INSTANCE
    mWidth = 1920;
    mHeight= 1080;
#else
    mWidth = pImageSize->width;
    mHeight= pImageSize->height;
#endif

    NPP_CHECK_NPP(nppiDCTInitAlloc(&pDCTState));
    cudaMalloc(&pdQuantizationTables, 64 * 4);
    memset(&oFrameHeader,       0,     sizeof(FrameHeader));
    memset(aQuantizationTables, 0, 4 * sizeof(QuantizationTable));
    memset(aHuffmanTables,      0, 4 * sizeof(HuffmanTable));

    //填充Huffman表
    aHuffmanTables[0].nClassAndIdentifier = 0;
    memcpy(aHuffmanTables[0].aCodes, STD_DC_Y_NRCODES,  16);
    memcpy(aHuffmanTables[0].aTable, STD_DC_Y_VALUES,   12);

    aHuffmanTables[1].nClassAndIdentifier = 1;
    memcpy(aHuffmanTables[1].aCodes, STD_DC_UV_NRCODES, 16);
    memcpy(aHuffmanTables[1].aTable, STD_DC_UV_VALUES,  12);

    aHuffmanTables[2].nClassAndIdentifier = 16;
    memcpy(aHuffmanTables[2].aCodes, STD_AC_Y_NRCODES,  16);
    memcpy(aHuffmanTables[2].aTable, STD_AC_Y_VALUES,  162);

    aHuffmanTables[3].nClassAndIdentifier = 17;
    memcpy(aHuffmanTables[3].aCodes, STD_AC_UV_NRCODES, 16);
    memcpy(aHuffmanTables[3].aTable, STD_AC_UV_VALUES, 162);

    //量化表。据说质量是在这里控制的。测试结果，可以看到文件大小变化，而质量感觉一般。
    aQuantizationTables[0].nPrecisionAndIdentifier = 0;
    //memcpy(aQuantizationTables[0].aTable, std_Y_QT, 64);
    SetQuality(aQuantizationTables[0].aTable, std_Y_QT,  quality);
    aQuantizationTables[1].nPrecisionAndIdentifier = 1;
    //memcpy(aQuantizationTables[1].aTable, std_UV_QT, 64);
    SetQuality(aQuantizationTables[1].aTable, std_UV_QT, quality);

    // Copy DCT coefficients and Quantization Tables from host to device 
    //拷贝时之字型扫描不可少，否则会出现一下人为畸变
    Npp8u aZigzag[] = {
         0,  1,  5,  6, 14, 15, 27, 28,
         2,  4,  7, 13, 16, 26, 29, 42,
         3,  8, 12, 17, 25, 30, 41, 43,
         9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63
    };

    //这个可以独立出来，不必每次初始化？
    for (int i = 0; i < 4; ++i)
    {
        Npp8u temp[64];

        for (int k = 0; k < 32; ++k)
        {
            temp[2 * k + 0] = aQuantizationTables[i].aTable[aZigzag[k + 0]];
            temp[2 * k + 1] = aQuantizationTables[i].aTable[aZigzag[k + 32]];
        }

        NPP_CHECK_CUDA(cudaMemcpyAsync((unsigned char *)pdQuantizationTables + i * 64, temp, 64, cudaMemcpyHostToDevice));
    }

    //这两句打开之后，画质很差。
    //NPP_CHECK_CUDA(cudaMemcpyAsync(pdQuantizationTables,      aQuantizationTables[0].aTable, 64, cudaMemcpyHostToDevice));
    //NPP_CHECK_CUDA(cudaMemcpyAsync(pdQuantizationTables + 64, aQuantizationTables[1].aTable, 64, cudaMemcpyHostToDevice));

    //必须是8？
    oFrameHeader.nSamplePrecision = 8;
    oFrameHeader.nComponents      = 3;
    oFrameHeader.aComponentIdentifier[0] = 1;
    oFrameHeader.aComponentIdentifier[1] = 2;
    oFrameHeader.aComponentIdentifier[2] = 3;
    oFrameHeader.aSamplingFactors[0] = 34;
    oFrameHeader.aSamplingFactors[1] = 17;
    oFrameHeader.aSamplingFactors[2] = 17;
    oFrameHeader.aQuantizationTableSelector[0] = 0;
    oFrameHeader.aQuantizationTableSelector[1] = 1;
    oFrameHeader.aQuantizationTableSelector[2] = 1;

    for (int i = 0; i < oFrameHeader.nComponents; ++i)
    {
        nMCUBlocksV = max(nMCUBlocksV, oFrameHeader.aSamplingFactors[i] & 0x0f);
        nMCUBlocksH = max(nMCUBlocksH, oFrameHeader.aSamplingFactors[i] >> 4);
    }

    /**
    * Allocates memory and creates a Huffman table in a format that is suitable for the encoder.
    */
    NppStatus t_status;
    t_status = nppiEncodeHuffmanSpecInitAlloc_JPEG(pHuffmanDCTables[0].aCodes, nppiDCTable, &apHuffmanDCTable[0]);
    t_status = nppiEncodeHuffmanSpecInitAlloc_JPEG(pHuffmanACTables[0].aCodes, nppiACTable, &apHuffmanACTable[0]);
    t_status = nppiEncodeHuffmanSpecInitAlloc_JPEG(pHuffmanDCTables[1].aCodes, nppiDCTable, &apHuffmanDCTable[1]);
    t_status = nppiEncodeHuffmanSpecInitAlloc_JPEG(pHuffmanACTables[1].aCodes, nppiACTable, &apHuffmanACTable[1]);

    //这里是1, 1
    t_status = nppiEncodeHuffmanSpecInitAlloc_JPEG(pHuffmanDCTables[1].aCodes, nppiDCTable, &apHuffmanDCTable[2]);
    t_status = nppiEncodeHuffmanSpecInitAlloc_JPEG(pHuffmanACTables[1].aCodes, nppiACTable, &apHuffmanACTable[2]);
    if (t_status)
    {
        //
    }


    oFrameHeader.nWidth  = mWidth;
    oFrameHeader.nHeight = mHeight;

    for (int i = 0; i < oFrameHeader.nComponents; ++i)
    {
        NppiSize oBlocks;
        NppiSize oBlocksPerMCU = { oFrameHeader.aSamplingFactors[i] >> 4, oFrameHeader.aSamplingFactors[i] & 0x0f };

        oBlocks.width = (int)ceil((oFrameHeader.nWidth   + 7) / 8 *
            static_cast<float>(oBlocksPerMCU.width) / nMCUBlocksH);
        oBlocks.width = DivUp(oBlocks.width, oBlocksPerMCU.width) * oBlocksPerMCU.width;

        oBlocks.height = (int)ceil((oFrameHeader.nHeight + 7) / 8 *
            static_cast<float>(oBlocksPerMCU.height) / nMCUBlocksV);
        oBlocks.height = DivUp(oBlocks.height, oBlocksPerMCU.height) * oBlocksPerMCU.height;

        aSrcSize[i].width  = oBlocks.width  * 8;
        aSrcSize[i].height = oBlocks.height * 8;

        // Allocate Memory
        size_t nPitch;
        NPP_CHECK_CUDA(cudaMallocPitch(&apdDCT[i], &nPitch, oBlocks.width * 64 * sizeof(Npp16s), oBlocks.height));
        aDCTStep[i] = static_cast<Npp32s>(nPitch);

        NPP_CHECK_CUDA(cudaMallocPitch(&apSrcImage[i], &nPitch, aSrcSize[i].width, aSrcSize[i].height));
        aSrcPitch[i] = nPitch;
        aSrcImageStep[i] = static_cast<Npp32s>(nPitch);
    }

    nScanSize = mWidth * mHeight * 2;
    nScanSize = nScanSize > (4 << 20) ? nScanSize : (4 << 20);
    NPP_CHECK_CUDA(cudaMalloc(&pdScan, nScanSize));

    NPP_CHECK_NPP(nppiEncodeHuffmanGetSize(aSrcSize[0], 3, &nTempSize));
    //这一句有内存泄露，38MB
    NPP_CHECK_CUDA(cudaMalloc(&pJpegEncoderTemp, nTempSize));

    pDstJpeg = new unsigned char[nScanSize];

    uint32_t nPitch = (mWidth + MEMORY_ALGN_DEVICE) & ~MEMORY_ALGN_DEVICE;
    mY = (uint8_t*)malloc(nPitch * mHeight);
    nPitch = (mWidth/2 + MEMORY_ALGN_DEVICE) & ~MEMORY_ALGN_DEVICE;
    mU = (uint8_t*)malloc(nPitch * mHeight/2);
    mV = (uint8_t*)malloc(nPitch * mHeight/2);
}

/*
brief:release host memory and device memory, no need always create the obj and release to induce the encoding performance
*/
void CudaJpegEncode::Release()
{
    if (pJpegEncoderTemp)
    {
        cudaFree(pJpegEncoderTemp);
        pJpegEncoderTemp = NULL;
    }
    if (pdQuantizationTables)
    {
        cudaFree(pdQuantizationTables);
        pdQuantizationTables = NULL;
    }
    if (pdScan)
    {
        cudaFree(pdScan);
        pdScan = NULL;
    }
    if (pDCTState)
    {
        nppiDCTFree(pDCTState);
        pDCTState = NULL;
    }
    if (pDstJpeg)
    {
        delete[] pDstJpeg;
        pDstJpeg = NULL;
    }
    if (mY)
    {
        free(mY);
        mY = NULL;
    }
    if (mU)
    {
        free(mU);
        mU = NULL;
    }
    if (mV)
    {
        free(mV);
        mV = NULL;
    }
    for (int i = 0; i < 3; ++i)
    {
        if (apdDCT[i])
        {
            cudaFree(apdDCT[i]);
            apdDCT[i] = NULL;
        }
        if (apSrcImage[i])
        {
            cudaFree(apSrcImage[i]);
            apSrcImage[i] = NULL;
        }
        if (apHuffmanDCTable[i])
        {
            nppiEncodeHuffmanSpecFree_JPEG(apHuffmanDCTable[i]);
            apHuffmanDCTable[i] = NULL;
        }
        if (apHuffmanACTable[i])
        {
            nppiEncodeHuffmanSpecFree_JPEG(apHuffmanACTable[i]);
            apHuffmanACTable[i] = NULL;
        }
    }
}


/*
brief: encode yuv data to jpeg format data, and write to file
*/
int CudaJpegEncode::EncodeJpeg(const IMAGE_DEST dest, char* pDest)
{
    int data_size = -1;

    oFrameHeader.nWidth  = mWidth;
    oFrameHeader.nHeight = mHeight;
    for (int i = 0; i < oFrameHeader.nComponents; ++i)
    {
        NppiSize oBlocks;
        NppiSize oBlocksPerMCU = { oFrameHeader.aSamplingFactors[i] >> 4, oFrameHeader.aSamplingFactors[i] & 0x0f };

        oBlocks.width  = (int)ceil((oFrameHeader.nWidth + 7) / 8 *
            static_cast<float>(oBlocksPerMCU.width) / nMCUBlocksH);
        oBlocks.width  = DivUp(oBlocks.width, oBlocksPerMCU.width) * oBlocksPerMCU.width;

        oBlocks.height = (int)ceil((oFrameHeader.nHeight + 7) / 8 *
            static_cast<float>(oBlocksPerMCU.height) / nMCUBlocksV);
        oBlocks.height = DivUp(oBlocks.height, oBlocksPerMCU.height) * oBlocksPerMCU.height;

        aSrcSize[i].width  = oBlocks.width  * 8;
        aSrcSize[i].height = oBlocks.height * 8;

        // Allocate Memory ??
        size_t nPitch;
        nPitch = (oBlocks.width * 64 * sizeof(Npp16s) + HD_MEMORY_ALGN_DEVICE) & ~HD_MEMORY_ALGN_DEVICE;
        aDCTStep[i] = static_cast<Npp32s>(nPitch);

        nPitch = (aSrcSize[i].width + HD_MEMORY_ALGN_DEVICE) & ~HD_MEMORY_ALGN_DEVICE;
        aSrcPitch[i] = nPitch;
        aSrcImageStep[i] = static_cast<Npp32s>(nPitch);
    }

    NPP_CHECK_CUDA(cudaMemcpy(apSrcImage[0], mY, aSrcPitch[0] * mHeight,     cudaMemcpyHostToDevice));
    NPP_CHECK_CUDA(cudaMemcpy(apSrcImage[1], mU, aSrcPitch[1] * mHeight / 2, cudaMemcpyHostToDevice));
    NPP_CHECK_CUDA(cudaMemcpy(apSrcImage[2], mV, aSrcPitch[2] * mHeight / 2, cudaMemcpyHostToDevice));

    /**
    * Forward DCT, quantization and level shift part of the JPEG encoding.
    * Input is expected in 8x8 macro blocks and output is expected to be in 64x1
    * macro blocks. The new version of the primitive takes the ROI in image pixel size and
    * works with DCT coefficients that are in zig-zag order.
    */
    int k = 0;

    NPP_CHECK_NPP(nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW(apSrcImage[0], aSrcImageStep[0],
        apdDCT[0], aDCTStep[0],
        pdQuantizationTables + k * 64,
        aSrcSize[0],
        pDCTState));

    k = 1;
    NPP_CHECK_NPP(nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW(apSrcImage[1], aSrcImageStep[1],
        apdDCT[1], aDCTStep[1],
        pdQuantizationTables + k * 64,
        aSrcSize[1],
        pDCTState));

    //k = 2; 加上之后，颜色不对。
    NPP_CHECK_NPP(nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW(apSrcImage[2], aSrcImageStep[2],
        apdDCT[2], aDCTStep[2],
        pdQuantizationTables + k * 64,
        aSrcSize[2],
        pDCTState));

    /**
    * Huffman Encoding of the JPEG Encoding.
    * Input is expected to be 64x1 macro blocks and output is expected as byte stuffed huffman encoded JPEG scan.
    */
    Npp32s nSs = 0;
    Npp32s nSe = 63;
    Npp32s nH  = 0;
    Npp32s nL  = 0;
    Npp32s nScanLength;

    NPP_CHECK_NPP(nppiEncodeHuffmanScan_JPEG_8u16s_P3R(apdDCT, aDCTStep,
        0, nSs, nSe, nH, nL,
        pdScan, &nScanLength,
        apHuffmanDCTable,
        apHuffmanACTable,
        aSrcSize,
        pJpegEncoderTemp));
    unsigned char *pDstOutput = pDstJpeg;

    writeMarker(0x0D8, pDstOutput);
    writeJFIFTag(pDstOutput);
    writeQuantizationTable(aQuantizationTables[0], pDstOutput);
    writeQuantizationTable(aQuantizationTables[1], pDstOutput);

    writeFrameHeader(oFrameHeader, pDstOutput);
    writeHuffmanTable(pHuffmanDCTables[0], pDstOutput);
    writeHuffmanTable(pHuffmanACTables[0], pDstOutput);
    writeHuffmanTable(pHuffmanDCTables[1], pDstOutput);
    writeHuffmanTable(pHuffmanACTables[1], pDstOutput);

    oScanHeader.nComponents           = 3;
    oScanHeader.aComponentSelector[0] = 1;
    oScanHeader.aComponentSelector[1] = 2;
    oScanHeader.aComponentSelector[2] = 3;
    oScanHeader.aHuffmanTablesSelector[0] = 0;
    oScanHeader.aHuffmanTablesSelector[1] = 17;
    oScanHeader.aHuffmanTablesSelector[2] = 17;
    oScanHeader.nSs = 0;
    oScanHeader.nSe = 63;
    oScanHeader.nA  = 0;

    writeScanHeader(oScanHeader, pDstOutput);
    NPP_CHECK_CUDA(cudaMemcpy(pDstOutput, pdScan, nScanLength, cudaMemcpyDeviceToHost));
    pDstOutput += nScanLength;
    writeMarker(0x0D9, pDstOutput);

    data_size = static_cast<int>(pDstOutput - pDstJpeg);
    if (dest == IMAGE_DEST_FILE)
    {
        // Write result to file.
        std::ofstream outputFile(pDest, ios::out | ios::binary);
        outputFile.write(reinterpret_cast<const char *>(pDstJpeg), data_size);
    }
    else
    {
    	//std::cout << " 1" << endl;
        memcpy((uint8_t*)pDest, (uint8_t*)pDstJpeg, data_size);
    }
    return data_size;
}

/*
brief: setting yuv image data, convert to yuv420
yuv_data:yuv data
yuv_fmt: yuv format, nv12 or nv21
w:       image width
h:       image height
size:    data size
*/
void CudaJpegEncode::SetData(const DataBuffer* pImageBuffer, int yuv_fmt, const RectSize* pImageSize)
{

    uint8_t* yuv_data = (uint8_t*)pImageBuffer->data;
    if (!yuv_data)
    {
        return;
    }

    int w = pImageSize->width;
    int h = pImageSize->height;

    mWidth  = w;
    mHeight = h;

    uint32_t    i, j;
    uint32_t    nPitch;
    uint32_t    off;
    uint32_t    off_yuv;
    uint32_t    half_h;
    uint32_t    half_w;
    uint32_t    u_size;
    uint8_t*    yuv_ptr;
    uint8_t*    u_ptr;
    uint8_t*    v_ptr;

    //从这一句来看，即使是同一种格式，进来也要处理一下。
    nPitch  = (w + HD_MEMORY_ALGN_DEVICE) & ~HD_MEMORY_ALGN_DEVICE;
    off     = 0;
    off_yuv = 0;
    for (i = 0; i < (uint32_t)h; i++)
    {
        memcpy(mY + off, yuv_data + off_yuv, w);
        off     += nPitch;
        off_yuv += w;
    }

    half_w = w >> 1;
    half_h = h >> 1;
    u_size = half_w * half_h;
    nPitch = (half_w + HD_MEMORY_ALGN_DEVICE) & ~HD_MEMORY_ALGN_DEVICE;
    switch (yuv_fmt)
    {
    case YUV_FMT_NV12:
    {
        off_yuv = w * h;
        off = 0;
        for (i = 0; i < half_h; i++)
        {  
            yuv_ptr = yuv_data + off_yuv;
            u_ptr = mU + off;
            v_ptr = mV + off;
            for (j = 0; j < (uint32_t)w; j += 2)
            {
                *u_ptr++ = *yuv_ptr++;
                *v_ptr++ = *yuv_ptr++;
            }
            off_yuv += w;
            off += nPitch;
        }
    }
    break;

    case YUV_FMT_NV21:
    {
        off_yuv = w * h;
        off = 0;
        for (i = 0; i < half_h; i++)
        {
            yuv_ptr = yuv_data + off_yuv;
            u_ptr = mU + off;
            v_ptr = mV + off;
            for (j = 0; j < (uint32_t)w; j += 2)
            {
                *v_ptr++ = *yuv_ptr++;
                *u_ptr++ = *yuv_ptr++;
            }
            off_yuv += w;
            off += nPitch;
        }
    }
    break;

    case YUV_FMT_YUV420:
    {
        off_yuv = w * h;
        off = 0;
        for (i = 0; i < half_h; i++)
        {
            memcpy(mU + off, yuv_data + off_yuv,          half_w);
            memcpy(mV + off, yuv_data + off_yuv + u_size, half_w);
            off_yuv += half_w;
            off += nPitch;
        }
    }
    break;

    default:
        break;
    }
}


//为什么要创新呢？
//因为这一句有内存泄露：NPP_CHECK_CUDA(cudaMalloc(&pJpegEncoderTemp, nTempSize));
#define MAX_CHANNELS 9
static CudaJpegEncode** g_pJpegEncoders = NULL;
int jpeg_npp_dest(const IMAGE_DEST dest, char *pDest, const int quality,
            const DataBuffer* pImageBuffer, const RectSize* pImageSize, const int channel)
{
#if USE_STATIC_INSTANCE
    if (g_pJpegEncoders == NULL)
    {
        int size = sizeof(CudaJpegEncode*)*MAX_CHANNELS;
        g_pJpegEncoders = (CudaJpegEncode**)malloc(size);
        memset(g_pJpegEncoders, 0, size);
    }
    if (g_pJpegEncoders[channel] == NULL)
    {
        g_pJpegEncoders[channel] = new CudaJpegEncode();
        g_pJpegEncoders[channel]->Init(quality, pImageSize);
    }
    g_pJpegEncoders[channel]->SetData(pImageBuffer, YUV_FMT_NV12, pImageSize);
    return g_pJpegEncoders[channel]->EncodeJpeg(dest, pDest);
#else
    CudaJpegEncode jpegEncoder;
    jpegEncoder.Init(quality, pImageSize);
    jpegEncoder.SetData(pImageBuffer, YUV_FMT_NV12, pImageSize);
    int size = jpegEncoder.EncodeJpeg(dest, pDest);
    jpegEncoder.Release();
    return size;
#endif
}

int jpeg_npp_file(const char *szOutputFile, const int quality,
            const DataBuffer* pImageBuffer, const RectSize* pImageSize, const int channel)
{
    return jpeg_npp_dest(IMAGE_DEST_FILE, (char*)szOutputFile, quality,
                         pImageBuffer, pImageSize, channel);
}

int jpeg_npp_mem(char *pJpegBuffer, const int quality,
            const DataBuffer* pImageBuffer, const RectSize* pImageSize, const int channel)
{
    return jpeg_npp_dest(IMAGE_DEST_MEM, pJpegBuffer, quality,
                         pImageBuffer, pImageSize, channel);
}

