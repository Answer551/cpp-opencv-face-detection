#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <iomanip>
#include <limits>
#ifdef _WIN32
#include <windows.h>
#endif
#include <nlohmann/json.hpp>

using namespace cv;
using namespace std;
using namespace std::filesystem;
using json = nlohmann::json;

Ptr<cv::FaceDetectorYN> detector = nullptr;

// 安全裁剪函数：手动计算边界，确保矩形完全在图像内部
Rect safeCrop(const Rect& r, const Size& imgSize) {
    int x = std::max(0, r.x);
    int y = std::max(0, r.y);
    int w = std::min(r.width,  imgSize.width  - x);
    int h = std::min(r.height, imgSize.height - y);
    if (w <= 0 || h <= 0) return Rect();
    return Rect(x, y, w, h);
}

VideoCapture open_camera() {
    const int backends[] = {CAP_MSMF, CAP_DSHOW, CAP_ANY};
    for (int index = 0; index <= 3; ++index) {
        for (int backend : backends) {
            VideoCapture cap(index, backend);
            if (cap.isOpened()) {
                cout << "[Camera] Opened device index " << index << endl;
                return cap;
            }
        }
    }
    return VideoCapture();
}

void test_camera() {
    VideoCapture cap = open_camera();
    if (!cap.isOpened()) {
        cerr << "[Camera] Access failed. Close apps using the camera and check Windows camera privacy settings." << endl;
        return;
    }
    cout << "[Camera] Preview opened. Press Q in the preview window to return." << endl;
    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cerr << "[Camera] Device opened but no frame was received." << endl;
            break;
        }
        putText(frame, "Camera OK - press Q to return", Point(20, 35),
                FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
        imshow("Camera Test", frame);
        int key = waitKey(10) & 0xff;
        if (key == 'q' || key == 'Q' || key == 27) break;
    }
    cap.release();
    destroyAllWindows();
}

bool init_yunet_detector(const Size& input_size = Size(640, 640)) {
    string model_path = "models/face_detection_yunet.onnx";
    if (!exists(model_path)) {
        cerr << "❌ 未找到 face_detection_yunet.onnx，请下载并放在项目根目录的 models/ 下" << endl;
        return false;
    }
    detector = cv::FaceDetectorYN::create(model_path, "", input_size, 0.7f, 0.3f, 5000);
    if (detector.empty()) {
        cerr << "❌ 创建 YuNet 检测器失败" << endl;
        return false;
    }
    return true;
}

map<int, string> load_name_map() {
    map<int, string> name_map;
    if (!exists("names.json")) return name_map;
    ifstream ifs("names.json");
    if (!ifs.is_open()) return name_map;
    json j;
    ifs >> j;
    for (auto& [key, val] : j.items()) {
        name_map[stoi(key)] = val.get<string>();
    }
    return name_map;
}

void save_name_map(const map<int, string>& name_map) {
    json j;
    for (auto& [id, name] : name_map) {
        j[to_string(id)] = name;
    }
    ofstream ofs("names.json");
    ofs << j.dump(4) << endl;
}

// ==================== 采集人脸数据 ====================
void collect_face_data(int face_id, const string& name) {
    if (detector.empty()) {
        cerr << "❌ 人脸检测器未初始化" << endl;
        return;
    }

    path dataset_path = "dataset";
    if (!exists(dataset_path)) create_directory(dataset_path);

    VideoCapture cap = open_camera();
    if (!cap.isOpened()) {
        cerr << "❌ 无法打开摄像头。请检查外接摄像头是否被 Windows 识别、USB 连接和相机权限。" << endl;
        return;
    }

    cout << "📸 正在为 " << name << " (ID: " << face_id << ") 采集，按 's' 保存，按 'q' 退出" << endl;
    int count = 0;

    Mat frame, face_roi;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // **关键：每帧都更新检测器输入尺寸，确保坐标与实际帧匹配**
        detector->setInputSize(Size(frame.cols, frame.rows));

        vector<Rect> faces;
        Mat results;
        detector->detect(frame, results);
        if (!results.empty()) {
            for (int i = 0; i < results.rows; ++i) {
                float* data = results.ptr<float>(i);
                int x = (int)data[0];
                int y = (int)data[1];
                int w = (int)data[2];
                int h = (int)data[3];
                // 只保留尺寸合理的人脸框，但坐标可能越界，后面再裁剪
                if (w > 50 && h > 50) {
                    faces.push_back(Rect(x, y, w, h));
                }
            }
        }

        face_roi = Mat();
        if (!faces.empty()) {
            Rect safeRect = safeCrop(faces[0], frame.size());
            if (!safeRect.empty()) {
                rectangle(frame, safeRect, Scalar(0, 255, 0), 2);
                face_roi = frame(safeRect);
            }
        }

        imshow("采集人脸 - 按s保存，按q退出", frame);
        char key = (char)waitKey(1);
        if (key == 's' && !face_roi.empty()) {
            Mat gray, resized;
            cvtColor(face_roi, gray, COLOR_BGR2GRAY);
            resize(gray, resized, Size(200, 200));
            string filename = dataset_path.string() + "/User." + to_string(face_id) + "." + to_string(count) + ".jpg";
            imwrite(filename, resized);
            cout << "✅ 已保存第 " << count + 1 << " 张" << endl;
            count++;
        } else if (key == 'q') {
            break;
        }
    }

    auto name_map = load_name_map();
    name_map[face_id] = name;
    save_name_map(name_map);
    cout << "✅ 已记录用户 " << name << " (ID: " << face_id << ")" << endl;

    cap.release();
    destroyAllWindows();
    cout << "🎉 采集完成！共 " << count << " 张图片" << endl;
}

// ==================== 训练模型（占位） ====================
void train_face_recognizer() {
    path dataset_path = "dataset";
    if (!exists(dataset_path)) {
        cerr << "❌ 未找到 dataset 文件夹，请先采集数据" << endl;
        return;
    }

    vector<Mat> faces;
    vector<int> labels;

    for (auto& entry : directory_iterator(dataset_path)) {
        if (!entry.is_regular_file()) continue;
        string filename = entry.path().filename().string();
        if (filename.substr(filename.find_last_of(".") + 1) != "jpg") continue;

        size_t pos1 = filename.find('.');
        if (pos1 == string::npos) continue;
        size_t pos2 = filename.find('.', pos1 + 1);
        if (pos2 == string::npos) continue;
        string id_str = filename.substr(pos1 + 1, pos2 - pos1 - 1);
        int face_id = stoi(id_str);

        Mat img = imread(entry.path().string(), IMREAD_GRAYSCALE);
        if (!img.empty()) {
            faces.push_back(img);
            labels.push_back(face_id);
        }
    }

    if (faces.empty()) {
        cerr << "❌ 没有有效图片，请先采集" << endl;
        return;
    }

    FileStorage model("trainer.yml", FileStorage::WRITE);
    model << "sample_count" << static_cast<int>(faces.size());
    model.release();
    cout << "✅ 训练完成！共处理 " << faces.size() << " 张图片，涉及 " << set<int>(labels.begin(), labels.end()).size() << " 个人" << endl;
}

// ==================== 实时识别 ====================
void realtime_recognition() {
    if (detector.empty()) {
        cerr << "❌ 人脸检测器未初始化" << endl;
        return;
    }
    if (!exists("names.json")) {
        cerr << "❌ 未找到 names.json，请先采集用户" << endl;
        return;
    }
    if (!exists("trainer.yml")) {
        cerr << "❌ 未找到 trainer.yml，请先训练模型" << endl;
        return;
    }

    auto name_map = load_name_map();
    vector<Mat> known_faces;
    vector<int> known_labels;
    for (const auto& entry : directory_iterator("dataset")) {
        if (!entry.is_regular_file() || entry.path().extension() != ".jpg") continue;
        string filename = entry.path().filename().string();
        size_t pos1 = filename.find('.');
        size_t pos2 = filename.find('.', pos1 + 1);
        if (pos1 == string::npos || pos2 == string::npos) continue;
        Mat image = imread(entry.path().string(), IMREAD_GRAYSCALE);
        if (!image.empty()) {
            known_faces.push_back(image);
            known_labels.push_back(stoi(filename.substr(pos1 + 1, pos2 - pos1 - 1)));
        }
    }
    if (known_faces.empty()) {
        cerr << "❌ 训练数据为空，请先采集并训练" << endl;
        return;
    }

    VideoCapture cap = open_camera();
    if (!cap.isOpened()) {
        cerr << "❌ 无法打开摄像头。请检查外接摄像头是否被 Windows 识别、USB 连接和相机权限。" << endl;
        return;
    }

    cout << "🔍 开始识别（YuNet + 模板匹配），按 'q' 退出" << endl;

    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // **关键：每帧都更新检测器输入尺寸**
        detector->setInputSize(Size(frame.cols, frame.rows));

        vector<Rect> faces;
        Mat results;
        detector->detect(frame, results);
        if (!results.empty()) {
            for (int i = 0; i < results.rows; ++i) {
                float* data = results.ptr<float>(i);
                int x = (int)data[0];
                int y = (int)data[1];
                int w = (int)data[2];
                int h = (int)data[3];
                if (w > 50 && h > 50) {
                    faces.push_back(Rect(x, y, w, h));
                }
            }
        }

        for (auto& rect : faces) {
            Rect safeRect = safeCrop(rect, frame.size());
            if (safeRect.empty()) continue;

            Mat face_roi = frame(safeRect);
            Mat gray, resized;
            cvtColor(face_roi, gray, COLOR_BGR2GRAY);
            resize(gray, resized, Size(200, 200));

            int label = -1;
            double confidence = 1e9;
            for (size_t i = 0; i < known_faces.size(); ++i) {
                Mat reference;
                resize(known_faces[i], reference, Size(200, 200));
                absdiff(resized, reference, reference);
                double score = mean(reference)[0];
                if (score < confidence) {
                    confidence = score;
                    label = known_labels[i];
                }
            }

            string display_name;
            Scalar color;
            if (confidence < 55 && label >= 0) {
                string name = name_map.count(label) ? name_map[label] : "Unknown";
                display_name = name + " (" + to_string(confidence) + ")";
                color = Scalar(0, 255, 0);
            } else {
                display_name = "Unknown (" + to_string(confidence) + ")";
                color = Scalar(0, 0, 255);
            }

            rectangle(frame, safeRect, color, 2);
            putText(frame, display_name, Point(safeRect.x, safeRect.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
        }

        imshow("人脸识别 (YuNet)", frame);
        if (waitKey(1) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
}

// ==================== 删除指定用户 ====================
void delete_user_data() {
    string uid_str;
    cout << "请输入要删除的用户ID（数字）: ";
    cin >> uid_str;
    if (!all_of(uid_str.begin(), uid_str.end(), ::isdigit)) {
        cerr << "❌ 请输入有效的数字ID" << endl;
        return;
    }
    int uid = stoi(uid_str);

    path dataset_path = "dataset";
    vector<path> photo_files;
    if (exists(dataset_path)) {
        for (auto& entry : directory_iterator(dataset_path)) {
            string fname = entry.path().filename().string();
            if (fname.find("User." + uid_str + ".") == 0 && fname.substr(fname.find_last_of(".") + 1) == "jpg") {
                photo_files.push_back(entry.path());
            }
        }
    }
    if (!photo_files.empty()) {
        for (auto& p : photo_files) {
            remove(p);
            cout << "🗑️ 已删除照片: " << p.string() << endl;
        }
        cout << "✅ 共删除 " << photo_files.size() << " 张照片" << endl;
    } else {
        cout << "❌ 未找到ID为 " << uid << " 的任何照片，请检查ID是否正确" << endl;
    }

    if (exists("names.json")) {
        auto name_map = load_name_map();
        if (name_map.count(uid)) {
            string deleted_name = name_map[uid];
            name_map.erase(uid);
            save_name_map(name_map);
            cout << "🗑️ 已从名字库中移除: " << deleted_name << " (ID: " << uid << ")" << endl;
        } else {
            cout << "ℹ️ names.json 中未找到 ID:" << uid << " 的名字记录" << endl;
        }
    }

    if (exists("trainer.yml")) {
        remove("trainer.yml");
        cout << "🗑️ 已删除旧模型文件 (trainer.yml)" << endl;
    }

    cout << "🎉 用户 " << uid << " 的数据已全部清除！" << endl;
    cout << "⚠️ 请务必重新运行【2. 训练模型】，否则程序将无法识别剩余用户！" << endl;
}

// ==================== 清空所有数据 ====================
void delete_all_data() {
    cout << "\n" << string(50, '=') << endl;
    cout << "⚠️  严 重 警 告 ⚠️" << endl;
    cout << "此操作将永久删除以下所有数据：" << endl;
    cout << "1. dataset 文件夹中的所有照片（所有用户）" << endl;
    cout << "2. trainer.yml 模型文件（训练好的特征库）" << endl;
    cout << "3. names.json 名字映射文件" << endl;
    cout << string(50, '=') << endl;

    string confirm;
    cout << "确认清空？请输入 'YES' (全大写) 以继续: ";
    cin >> confirm;
    if (confirm != "YES") {
        cout << "❌ 操作已取消，数据安全保留。" << endl;
        return;
    }

    if (exists("dataset")) {
        remove_all("dataset");
        cout << "✅ 已删除 'dataset' 文件夹及其中的所有照片" << endl;
    } else {
        cout << "ℹ️ 'dataset' 文件夹不存在，跳过" << endl;
    }
    create_directory("dataset");
    cout << "✅ 已重新创建空的 'dataset' 文件夹" << endl;

    if (exists("trainer.yml")) {
        remove("trainer.yml");
        cout << "✅ 已删除 'trainer.yml' 模型文件" << endl;
    } else {
        cout << "ℹ️ 'trainer.yml' 不存在，跳过" << endl;
    }

    if (exists("names.json")) {
        remove("names.json");
        cout << "✅ 已删除 'names.json' 名字映射文件" << endl;
    } else {
        cout << "ℹ️ 'names.json' 不存在，跳过" << endl;
    }

    cout << "\n" << string(50, '=') << endl;
    cout << "🎉 重置完成！系统已恢复至初始状态，请从【采集人脸】重新开始。" << endl;
    cout << string(50, '=') << endl;
}

// ==================== 查询所有用户 ====================
void list_all_users() {
    if (!exists("names.json")) {
        cout << "[Query] No users yet. names.json does not exist." << endl;
        return;
    }
    auto name_map = load_name_map();
    if (name_map.empty()) {
        cout << "[Query] User list is empty." << endl;
        return;
    }

    cout << "\n" << string(50, '=') << endl;
    cout << "        📋 当前用户列表" << endl;
    cout << string(50, '=') << endl;
    cout << left << setw(6) << "ID" << setw(10) << "姓名" << setw(8) << "照片数" << endl;
    cout << string(50, '-') << endl;

    int total_photos = 0;
    vector<int> zero_users;
    for (auto& [uid, name] : name_map) {
        int count = 0;
        if (exists("dataset")) {
            for (auto& entry : directory_iterator("dataset")) {
                string fname = entry.path().filename().string();
                if (fname.find("User." + to_string(uid) + ".") == 0 && fname.substr(fname.find_last_of(".") + 1) == "jpg")
                    count++;
            }
        }
        total_photos += count;
        cout << left << setw(6) << uid << setw(10) << name << setw(8) << count << endl;
        if (count == 0) zero_users.push_back(uid);
    }
    cout << string(50, '-') << endl;
    cout << "👥 总用户数: " << name_map.size() << " 人" << endl;
    cout << "📸 总照片数: " << total_photos << " 张" << endl;
    cout << string(50, '=') << endl;

    if (!zero_users.empty()) {
        cout << "⚠️ 警告：以下用户没有照片文件，请重新采集: ";
        for (int id : zero_users) cout << id << " ";
        cout << endl;
    }
}

void rename_user() {
    int uid;
    string new_name;
    cout << "[Modify] User ID: ";
    if (!(cin >> uid)) {
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        cout << "[Modify] Invalid ID." << endl;
        return;
    }
    cout << "[Modify] New name: ";
    cin >> new_name;
    auto name_map = load_name_map();
    auto it = name_map.find(uid);
    if (it == name_map.end()) {
        cout << "[Modify] User not found." << endl;
        return;
    }
    it->second = new_name;
    save_name_map(name_map);
    cout << "[Modify] User updated." << endl;
}

// ==================== 主菜单 ====================
int main() {
#ifdef _WIN32
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
    wchar_t executable_buffer[MAX_PATH];
    DWORD executable_length = GetModuleFileNameW(nullptr, executable_buffer, MAX_PATH);
    if (executable_length > 0) {
        path executable_dir = path(executable_buffer).parent_path();
        string dir_name = executable_dir.filename().string();
        if (dir_name == "build" || dir_name == "tast") {
            current_path(executable_dir.parent_path());
        }
    }
#endif
    bool detector_ready = false;

    while (true) {
        cout << "\n========================================" << endl;
        cout << "      OpenCV Face Recognition System" << endl;
        cout << "========================================" << endl;
        cout << "1. 创建用户 / 采集人脸" << endl;
        cout << "2. 训练模型" << endl;
        cout << "3. 实时识别" << endl;
        cout << "4. 退出" << endl;
        cout << "5. 删除用户" << endl;
        cout << "6. 删除所有数据" << endl;
        cout << "7. 查询用户" << endl;
        cout << "8. 修改用户名" << endl;
        cout << "9. 测试摄像头" << endl;
        cout << "========================================" << endl;
        cout << "选择一个选项: ";

        int choice;
        cin >> choice;
        if (cin.fail()) {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cerr << "无效输入，请重新运行" << endl;
            continue;
        }

        switch (choice) {
            case 1: {
                if (!detector_ready) detector_ready = init_yunet_detector();
                if (!detector_ready) break;
                int uid;
                string name;
                cout << "请输入用户ID（数字，例如 10086）: ";
                cin >> uid;
                cout << "请输入用户姓名（例如 张三）: ";
                cin >> name;
                collect_face_data(uid, name);
                break;
            }
            case 2:
                train_face_recognizer();
                break;
            case 3:
                if (!detector_ready) detector_ready = init_yunet_detector();
                if (!detector_ready) break;
                realtime_recognition();
                break;
            case 4:
                cout << "退出程序" << endl;
                return 0;
            case 5:
                delete_user_data();
                break;
            case 6:
                delete_all_data();
                break;
            case 7:
                list_all_users();
                break;
            case 8:
                rename_user();
                break;
            case 9:
                test_camera();
                break;
            default:
                cerr << "无效输入，请重新运行" << endl;
                break;
        }
    }
    return 0;
}
