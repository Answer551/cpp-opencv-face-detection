#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/objdetect/face.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

#include <nlohmann/json.hpp>

using cv::Mat;
using cv::Ptr;
using cv::Rect;
using cv::Scalar;
using cv::Size;
using cv::VideoCapture;
using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

constexpr float kDetectionScore = 0.75F;
constexpr double kCosineThreshold = 0.40;
const fs::path kDatasetDir = "dataset_sface";
const fs::path kNamesFile = "names_sface.json";
const fs::path kGalleryFile = "sface_gallery.yml";

Ptr<cv::FaceDetectorYN> detector;
Ptr<cv::FaceRecognizerSF> recognizer;

VideoCapture openCamera() {
    const int backends[] = {cv::CAP_MSMF, cv::CAP_DSHOW, cv::CAP_ANY};
    for (int index = 0; index <= 3; ++index) {
        for (int backend : backends) {
            VideoCapture cap(index, backend);
            if (cap.isOpened()) {
                std::cout << "[Camera] Opened device " << index << std::endl;
                return cap;
            }
        }
    }
    return VideoCapture();
}

bool initModels() {
    const std::string detectorPath = "models/face_detection_yunet.onnx";
    const std::string recognizerPath = "models/face_recognition_sface_2021dec.onnx";
    if (!fs::exists(detectorPath) || !fs::exists(recognizerPath)) {
        std::cerr << "[Model] Missing YuNet or SFace ONNX model in models/." << std::endl;
        return false;
    }
    detector = cv::FaceDetectorYN::create(detectorPath, "", Size(640, 640), kDetectionScore, 0.3F, 5000);
    recognizer = cv::FaceRecognizerSF::create(recognizerPath, "");
    return !detector.empty() && !recognizer.empty();
}

std::map<int, std::string> loadNames() {
    std::map<int, std::string> names;
    std::ifstream input(kNamesFile);
    if (!input.is_open()) return names;
    json document;
    input >> document;
    for (auto& [id, name] : document.items()) names[std::stoi(id)] = name.get<std::string>();
    return names;
}

void saveNames(const std::map<int, std::string>& names) {
    json document;
    for (const auto& [id, name] : names) document[std::to_string(id)] = name;
    std::ofstream(kNamesFile) << document.dump(2) << std::endl;
}

bool bestFace(const Mat& frame, Mat& faceBox) {
    detector->setInputSize(frame.size());
    Mat faces;
    detector->detect(frame, faces);
    if (faces.empty()) return false;

    int bestIndex = 0;
    float bestScore = faces.at<float>(0, 14);
    for (int row = 1; row < faces.rows; ++row) {
        const float score = faces.at<float>(row, 14);
        if (score > bestScore) {
            bestScore = score;
            bestIndex = row;
        }
    }
    faceBox = faces.row(bestIndex).clone();
    return true;
}

bool alignedFeature(const Mat& frame, const Mat& faceBox, Mat& aligned, Mat& feature) {
    try {
        recognizer->alignCrop(frame, faceBox, aligned);
        recognizer->feature(aligned, feature);
        feature = feature.reshape(1, 1).clone();
        cv::normalize(feature, feature);
        return !feature.empty();
    } catch (const cv::Exception& error) {
        std::cerr << "[SFace] " << error.what() << std::endl;
        return false;
    }
}

int parseId(const fs::path& imagePath) {
    const std::string name = imagePath.filename().string();
    const size_t first = name.find('.');
    const size_t second = name.find('.', first + 1);
    if (first == std::string::npos || second == std::string::npos) return -1;
    try {
        return std::stoi(name.substr(first + 1, second - first - 1));
    } catch (...) {
        return -1;
    }
}

bool loadGallery(std::map<int, Mat>& gallery) {
    cv::FileStorage storage(kGalleryFile.string(), cv::FileStorage::READ);
    if (!storage.isOpened()) return false;
    Mat features;
    storage["features"] >> features;
    cv::FileNode ids = storage["ids"];
    if (features.empty() || ids.empty() || static_cast<int>(ids.size()) != features.rows) return false;
    int row = 0;
    for (auto it = ids.begin(); it != ids.end(); ++it, ++row) {
        gallery[static_cast<int>(*it)] = features.row(row).clone();
    }
    return !gallery.empty();
}

void collectSamples() {
    int id = 0;
    std::string name;
    std::cout << "ID: "; std::cin >> id;
    std::cout << "Name (no spaces): "; std::cin >> name;
    fs::create_directories(kDatasetDir);

    VideoCapture camera = openCamera();
    if (!camera.isOpened()) {
        std::cerr << "[Camera] Unable to open camera." << std::endl;
        return;
    }
    std::cout << "Press S to save an aligned face, Q to finish." << std::endl;
    int saved = 0;
    Mat frame;
    while (true) {
        camera >> frame;
        if (frame.empty()) break;
        Mat faceBox, aligned, feature;
        const bool hasFace = bestFace(frame, faceBox) && alignedFeature(frame, faceBox, aligned, feature);
        if (hasFace) {
            Rect box(static_cast<int>(faceBox.at<float>(0, 0)), static_cast<int>(faceBox.at<float>(0, 1)),
                     static_cast<int>(faceBox.at<float>(0, 2)), static_cast<int>(faceBox.at<float>(0, 3)));
            box &= Rect(0, 0, frame.cols, frame.rows);
            if (box.area() > 0) cv::rectangle(frame, box, Scalar(0, 255, 0), 2);
            cv::putText(frame, "Face aligned: press S", {20, 35}, cv::FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
        } else {
            cv::putText(frame, "No reliable face", {20, 35}, cv::FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
        }
        cv::imshow("SFace sample collection", frame);
        const int key = cv::waitKey(1) & 0xff;
        if ((key == 's' || key == 'S') && hasFace) {
            const fs::path file = kDatasetDir / ("User." + std::to_string(id) + "." + std::to_string(saved++) + ".png");
            cv::imwrite(file.string(), aligned);
            std::cout << "[Saved] " << file.string() << std::endl;
        }
        if (key == 'q' || key == 'Q' || key == 27) break;
    }
    camera.release();
    cv::destroyAllWindows();
    auto names = loadNames();
    names[id] = name;
    saveNames(names);
    std::cout << "[Collect] Saved " << saved << " aligned samples. Capture 10-20 varied samples per person." << std::endl;
}

void buildGallery() {
    if (!fs::exists(kDatasetDir)) {
        std::cerr << "[Gallery] No collected samples. Use option 1 first." << std::endl;
        return;
    }
    std::map<int, Mat> sums;
    std::map<int, int> counts;
    for (const auto& entry : fs::directory_iterator(kDatasetDir)) {
        if (!entry.is_regular_file()) continue;
        const int id = parseId(entry.path());
        if (id < 0) continue;
        Mat image = cv::imread(entry.path().string());
        if (image.empty()) continue;
        Mat feature;
        recognizer->feature(image, feature);
        feature = feature.reshape(1, 1).clone();
        cv::normalize(feature, feature);
        if (sums.find(id) == sums.end()) sums[id] = Mat::zeros(feature.size(), feature.type());
        sums[id] += feature;
        ++counts[id];
    }
    if (sums.empty()) {
        std::cerr << "[Gallery] No valid aligned samples found." << std::endl;
        return;
    }
    std::vector<int> ids;
    std::vector<Mat> rows;
    for (auto& [id, featureSum] : sums) {
        featureSum /= static_cast<float>(counts[id]);
        cv::normalize(featureSum, featureSum);
        ids.push_back(id);
        rows.push_back(featureSum);
        std::cout << "[Gallery] ID " << id << ": " << counts[id] << " samples" << std::endl;
    }
    Mat features;
    cv::vconcat(rows, features);
    cv::FileStorage storage(kGalleryFile.string(), cv::FileStorage::WRITE);
    storage << "ids" << "[";
    for (const int id : ids) storage << id;
    storage << "]";
    storage << "features" << features;
    std::cout << "[Gallery] Wrote " << kGalleryFile.string() << " with " << ids.size() << " identities." << std::endl;
}

void recognizeRealtime() {
    std::map<int, Mat> gallery;
    if (!loadGallery(gallery)) {
        std::cerr << "[Recognize] Gallery missing. Use option 2 after collecting samples." << std::endl;
        return;
    }
    const auto names = loadNames();
    VideoCapture camera = openCamera();
    if (!camera.isOpened()) return;
    std::cout << "[Recognize] Q closes preview. Threshold=" << kCosineThreshold << std::endl;
    Mat frame;
    while (true) {
        camera >> frame;
        if (frame.empty()) break;
        detector->setInputSize(frame.size());
        Mat faces;
        detector->detect(frame, faces);
        for (int row = 0; row < faces.rows; ++row) {
            Mat aligned, feature;
            if (!alignedFeature(frame, faces.row(row), aligned, feature)) continue;
            int bestId = -1;
            double bestScore = -1.0;
            for (const auto& [id, reference] : gallery) {
                const double score = recognizer->match(feature, reference, cv::FaceRecognizerSF::FR_COSINE);
                if (score > bestScore) { bestScore = score; bestId = id; }
            }
            Rect box(static_cast<int>(faces.at<float>(row, 0)), static_cast<int>(faces.at<float>(row, 1)),
                     static_cast<int>(faces.at<float>(row, 2)), static_cast<int>(faces.at<float>(row, 3)));
            box &= Rect(0, 0, frame.cols, frame.rows);
            const bool known = bestScore >= kCosineThreshold && names.find(bestId) != names.end();
            const Scalar color = known ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
            const std::string label = (known ? names.at(bestId) : "Unknown") + " " + cv::format("%.3f", bestScore);
            if (box.area() > 0) {
                cv::rectangle(frame, box, color, 2);
                cv::putText(frame, label, {box.x, std::max(25, box.y - 8)}, cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
            }
        }
        cv::imshow("YuNet + SFace identity matching", frame);
        const int key = cv::waitKey(1) & 0xff;
        if (key == 'q' || key == 'Q' || key == 27) break;
    }
    camera.release();
    cv::destroyAllWindows();
}

void listUsers() {
    const auto names = loadNames();
    if (names.empty()) { std::cout << "[Users] No identities collected." << std::endl; return; }
    for (const auto& [id, name] : names) std::cout << "ID=" << id << " name=" << name << std::endl;
}

}  // namespace

int main() {
#ifdef _WIN32
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
#endif
    if (!initModels()) return 1;
    while (true) {
        std::cout << "\n=== YuNet + SFace Face Identity ===\n"
                  << "1. Collect aligned samples\n"
                  << "2. Build feature gallery\n"
                  << "3. Real-time recognition\n"
                  << "4. List users\n"
                  << "5. Exit\nChoice: ";
        int choice = 0;
        if (!(std::cin >> choice)) return 1;
        if (choice == 1) collectSamples();
        else if (choice == 2) buildGallery();
        else if (choice == 3) recognizeRealtime();
        else if (choice == 4) listUsers();
        else if (choice == 5) return 0;
        else std::cout << "Invalid choice." << std::endl;
    }
}
