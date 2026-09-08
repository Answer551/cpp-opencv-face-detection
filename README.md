# C++ OpenCV Face Detection and Local Identity Matching

A Windows C++17 project that demonstrates two local camera identity-matching pipelines built with OpenCV DNN and CMake.

The recommended pipeline is `YuNet + SFace`:

```text
camera frame
  -> YuNet face detection with 5 landmarks
  -> SFace alignCrop landmark alignment
  -> SFace feature embedding
  -> cosine similarity against a local feature gallery
  -> known identity or Unknown
```

It is a local demonstration project, not a production access-control system. Local images and feature galleries are intentionally ignored by Git.

## What Is Included

- `face_identity_tool`: the original YuNet detector plus grayscale pixel-template matching prototype.
- `face_identity_sface`: the upgraded YuNet + SFace pipeline. It stores aligned samples, averages one normalized SFace feature vector per person, and uses a cosine-similarity threshold to return an identity or `Unknown`.
- JSON files for the local `ID -> name` mapping.
- CMake build configuration and OpenCV runtime-DLL copying for Windows/MinGW.

## Why SFace Instead of Pixel Templates

The original program compares resized grayscale pixels. It is easy to understand but highly sensitive to lighting, distance and head pose.

The SFace program extracts a learned feature vector from an aligned face. The matching stage compares feature direction using cosine similarity, which is generally more robust to normal appearance changes. This is still a fixed pre-trained model plus a local feature gallery; it does not train a new deep face-recognition network.

## Models

The repository tracks the small YuNet detector. Download the SFace recognition model separately and put it here:

```text
models/face_recognition_sface_2021dec.onnx
```

Official source: <https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface>

The SFace model is ignored by Git so the repository remains lightweight.

## Build

Requirements:

- Windows 10/11
- CMake 3.20+
- Ninja and a MinGW C++17 compiler
- OpenCV 4.10 built for the same MinGW toolchain, including `objdetect` and `dnn`

This machine uses an OpenCV installation placed under `third_party/opencv-mingw`. Configure and build from the project root:

```powershell
cmake -S . -B build-sface -G Ninja `
  -DOpenCV_DIR="D:/work/cpp-opencv-face-detection/third_party/opencv-mingw" `
  -DOPENCV_RUNTIME_DIR="D:/work/cpp-opencv-face-detection/third_party/opencv-mingw/x64/mingw/bin"
cmake --build build-sface
```

Run the upgraded executable:

```powershell
.\build-sface\face_identity_sface.exe
```

## Using the SFace Pipeline

1. Select `1` and capture 10-20 samples for each person. Move slightly between captures and include ordinary lighting variation. The program saves landmark-aligned images under `dataset_sface/`.
2. Select `2` to build `sface_gallery.yml`. It averages and L2-normalizes each person's SFace feature vectors.
3. Select `3` for real-time matching. The display shows a name only when the best cosine similarity is at least `0.40`; otherwise it displays `Unknown`.
4. Adjust the threshold only after collecting separate validation photos. A lower threshold reduces false unknowns but increases the risk of matching an unregistered person to a known identity.

## Validation Checklist

Do not evaluate using only the images used to build the gallery. For each enrolled person, keep several different photos aside for testing. Test at least:

- known identity under normal lighting;
- known identity with moderate pose and distance changes;
- an unregistered person, which should return `Unknown`;
- no-face frames and multiple-face frames.

Record detection latency, feature-extraction latency, matching latency, false matches and false unknowns before making any performance claim.

## Project Boundaries

- YuNet detects faces and provides five landmarks; it does not identify a person by itself.
- SFace produces features; the local gallery and cosine threshold make the final identity decision.
- The default threshold is only a starting point, not a universal security threshold.
- This project does not upload face data, but user consent and data-protection requirements still apply in any real deployment.
