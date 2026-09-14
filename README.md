# C++ OpenCV 人脸检测与本地身份匹配

这是一个面向 Windows 的 C++17 示例项目，使用 OpenCV DNN 和 CMake 实现两套本地摄像头身份匹配流程。推荐使用 **YuNet + SFace** 流程：

```text
摄像头画面
  -> YuNet 人脸检测与 5 点关键点定位
  -> SFace alignCrop 关键点对齐
  -> SFace 人脸特征提取
  -> 与本地特征库计算余弦相似度
  -> 输出已登记身份或 Unknown
```

本项目用于本地功能演示，不是可直接投入生产的门禁系统。为保护隐私并控制仓库体积，本地人脸图片和特征库不会提交到 Git。

## 项目内容

- `face_identity_tool`：原始版本，使用 YuNet 检测人脸，并通过灰度像素模板完成身份匹配。
- `face_identity_sface`：升级版本，使用 YuNet + SFace。程序保存对齐后的人脸样本，为每个人计算并平均一个归一化的 SFace 特征向量，再通过余弦相似度阈值输出姓名或 `Unknown`。
- 使用 JSON 文件保存本地 `ID -> 姓名` 映射。
- 提供 CMake 构建配置，并在 Windows/MinGW 环境下自动复制 OpenCV 运行时 DLL。

## 为什么使用 SFace 替代像素模板

原始程序直接比较缩放后的灰度像素，原理直观，但对光照、距离和头部姿态非常敏感。SFace 版本先对齐人脸，再提取预训练模型学习到的特征向量，最后使用余弦相似度比较特征方向，因此通常能更好地适应常见的外观变化。

需要注意：本项目使用固定的预训练模型和本地特征库，并没有重新训练一个深度人脸识别网络。

## 模型准备

仓库已包含体积较小的 YuNet 人脸检测模型。请单独下载 SFace 人脸识别模型，并放到以下位置：

```text
models/face_recognition_sface_2021dec.onnx
```

官方来源：<https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface>

SFace 模型已被 `.gitignore` 忽略，以保持仓库轻量。

## 构建

环境要求：

- Windows 10/11
- CMake 3.20+
- Ninja 和支持 C++17 的 MinGW 编译器
- 使用同一套 MinGW 工具链编译的 OpenCV 4.10，并包含 `objdetect` 和 `dnn` 模块

本机使用的 OpenCV 位于 `third_party/opencv-mingw`。在项目根目录执行：

```powershell
cmake -S . -B build-sface -G Ninja `
  -DOpenCV_DIR="D:/work/cpp-opencv-face-detection/third_party/opencv-mingw" `
  -DOPENCV_RUNTIME_DIR="D:/work/cpp-opencv-face-detection/third_party/opencv-mingw/x64/mingw/bin"
cmake --build build-sface
```

运行升级后的程序：

```powershell
.\build-sface\face_identity_sface.exe
```

## 使用 SFace 流程

1. 选择 `1`，为每个人采集 10～20 张样本。采集过程中可轻微移动，并覆盖日常光照变化。程序会将关键点对齐后的人脸图片保存到 `dataset_sface/`。
2. 选择 `2`，生成 `sface_gallery.yml`。程序会计算每个人的 SFace 特征均值，并进行 L2 归一化。
3. 选择 `3`，进行实时身份匹配。只有最佳余弦相似度不低于 `0.40` 时才显示姓名，否则显示 `Unknown`。
4. 不要凭感觉修改阈值。应先额外采集独立验证照片，再根据结果调整。降低阈值可以减少已登记人员被判为未知的情况，但会增加未登记人员被误认为已知身份的风险。

## 验证清单

不要只使用构建特征库时的图片进行评估。应为每位已登记人员额外保留若干测试照片，至少覆盖：

- 正常光照下的已登记人员；
- 存在一定姿态和距离变化的已登记人员；
- 未登记人员，预期输出 `Unknown`；
- 画面中无人脸以及存在多张人脸的情况。

在对性能作出结论前，应分别记录人脸检测延迟、特征提取延迟、身份匹配延迟、误匹配次数和误判为未知的次数。

## 项目边界

- YuNet 负责检测人脸并输出 5 个关键点，本身不能判断人物身份。
- SFace 负责生成特征；最终身份由本地特征库和余弦相似度阈值共同决定。
- 默认阈值只用于提供起点，并不是适用于所有场景的安全阈值。
- 本项目不会上传人脸数据，但在真实部署中仍需获得相关人员同意，并遵守数据保护要求。
