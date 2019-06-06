## rtsp硬件解码库

### 关于

通过ffmpeg和cuda 对rtsp视频流或本地视频进行硬件解码,作为人脸识别项目的数据采集模块

---

### 依赖

- ffmpeg-4.1.3
- cuda-10.1(完整安装,包含samples目录)
- opencv-3.4.1
- pybinder11

---

### 构建

首先确保makefile文件与本机依赖环境一致

```
cd src 
cd Release
make clean
make
```

---

### 测试

提供的接口和使用方法见test文件夹