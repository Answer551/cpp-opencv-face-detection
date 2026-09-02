# C++ OpenCV Face Detection

一个使用 C++17 和 OpenCV DNN 加载 YuNet ONNX 模型的本地摄像头人脸检测与身份管理练习项目。

## Scope

- 使用 YuNet 检测摄像头中的人脸。
- 采集本地人脸图片，使用 JSON 维护 ID 到姓名的映射。
- 身份匹配使用灰度图模板差异，不是深度学习人脸识别模型。
- 支持用户查询、改名和删除本地数据。

## Prerequisites

- Windows 10/11
- CMake 3.20+
- Ninja and a MinGW C++17 compiler
- OpenCV 4.10 C++ development package, including `OpenCVConfig.cmake` and runtime DLLs

## Build

Configure the project with the local OpenCV installation path. On this computer:

```powershell
cmake -S . -B build -G Ninja `
  -DOpenCV_DIR="C:/Users/Administrator/Desktop/DEMO1/third_party/opencv-mingw" `
  -DOPENCV_RUNTIME_DIR="C:/Users/Administrator/Desktop/DEMO1/third_party/opencv-mingw/bin"
cmake --build build
```

Run the executable from the project root:

```powershell
.\build\face_identity_tool.exe
```

Choose `9` to test the camera first. The program uses `Q` or `Esc` to close camera previews.

## Layout

```text
src/main.cpp                    # Menu, YuNet detection and local template matching
models/face_detection_yunet.onnx
third_party/nlohmann/json.hpp
dataset/                        # Local photos generated at runtime; ignored by Git
```

## Limitations

The matching logic is a simple pixel-difference template comparison. It is sensitive to pose, lighting and camera distance, and should not be presented as a production-grade face recognition system. The project demonstrates C++ OpenCV DNN inference, camera I/O, local persistence and CMake-based builds.
