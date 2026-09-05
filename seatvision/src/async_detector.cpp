#include "seatvision/async_detector.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace seatvision {
namespace {
constexpr int kInputSize = 640;
constexpr std::array<const char*, 80> kCocoNames = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

ObjectKind kind_for(int class_id) {
    if (class_id == 0) return ObjectKind::Person;
    if (class_id == 56) return ObjectKind::Chair;
    if (class_id == 24) return ObjectKind::Backpack;
    if (class_id == 39) return ObjectKind::Bottle;
    return ObjectKind::Other;
}
}  // namespace

AsyncYoloDetector::AsyncYoloDetector(std::string model_path, float confidence_threshold, float nms_threshold)
    : model_path_(std::move(model_path)), confidence_threshold_(confidence_threshold), nms_threshold_(nms_threshold) {
    ready_ = std::filesystem::is_regular_file(model_path_);
    if (!ready_) std::cerr << "Model not found: " << model_path_ << " (running camera/state-only mode)\n";
    worker_ = std::thread(&AsyncYoloDetector::worker, this);
}

AsyncYoloDetector::~AsyncYoloDetector() {
    { std::lock_guard lock(mutex_); stop_ = true; }
    work_ready_.notify_one();
    if (worker_.joinable()) worker_.join();
}

bool AsyncYoloDetector::ready() const { return ready_; }

void AsyncYoloDetector::submit(const cv::Mat& frame) {
    if (!ready_ || frame.empty()) return;
    { std::lock_guard lock(mutex_); pending_frame_ = frame.clone(); }
    work_ready_.notify_one();
}

std::optional<FrameResult> AsyncYoloDetector::latest() {
    std::lock_guard lock(mutex_);
    return latest_result_;
}

void AsyncYoloDetector::worker() {
    if (!ready_) return;
    try {
        bool reported_first_result = false;
        for (;;) {
            cv::Mat frame;
            {
                std::unique_lock lock(mutex_);
                work_ready_.wait(lock, [&] { return stop_ || pending_frame_.has_value(); });
                if (stop_) return;
                frame = std::move(*pending_frame_);
                pending_frame_.reset();
            }
            auto result = infer(frame);
            if (!reported_first_result) {
                std::cerr << "YOLO inference active (" << result.detections.size() << " detections in first frame)\n";
                reported_first_result = true;
            }
            { std::lock_guard lock(mutex_); latest_result_ = std::move(result); }
        }
    } catch (const cv::Exception& error) {
        std::cerr << "YOLO detector stopped: " << error.what() << '\n';
        ready_ = false;
    }
}

FrameResult AsyncYoloDetector::infer(const cv::Mat& frame) const {
    thread_local cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path_);
    const float scale = std::min(static_cast<float>(kInputSize) / frame.cols, static_cast<float>(kInputSize) / frame.rows);
    const int resized_w = static_cast<int>(std::round(frame.cols * scale));
    const int resized_h = static_cast<int>(std::round(frame.rows * scale));
    const int pad_x = (kInputSize - resized_w) / 2;
    const int pad_y = (kInputSize - resized_h) / 2;
    cv::Mat letterboxed(kInputSize, kInputSize, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::resize(frame, letterboxed(cv::Rect(pad_x, pad_y, resized_w, resized_h)), cv::Size(resized_w, resized_h));
    net.setInput(cv::dnn::blobFromImage(letterboxed, 1.0 / 255.0, cv::Size(kInputSize, kInputSize), cv::Scalar(), true, false));
    cv::Mat output = net.forward();
    cv::Mat rows;
    if (output.dims == 3 && output.size[1] == 84) {
        // YOLOv8 exports [batch, attributes(84), candidates(8400)].
        // Transpose to one candidate per row before class scoring.
        cv::transpose(cv::Mat(output.size[1], output.size[2], CV_32F, output.ptr<float>()), rows);
    }
    else rows = output.reshape(1, output.total() / 84);

    std::vector<cv::Rect> boxes; std::vector<float> scores; std::vector<int> classes;
    for (int r = 0; r < rows.rows; ++r) {
        const float* data = rows.ptr<float>(r);
        int class_id = static_cast<int>(std::max_element(data + 4, data + 84) - (data + 4));
        float score = data[4 + class_id];
        if (score < confidence_threshold_) continue;
        float x = (data[0] - data[2] / 2 - pad_x) / scale;
        float y = (data[1] - data[3] / 2 - pad_y) / scale;
        float w = data[2] / scale, h = data[3] / scale;
        cv::Rect box(cvRound(x), cvRound(y), cvRound(w), cvRound(h));
        box &= cv::Rect(0, 0, frame.cols, frame.rows);
        if (box.area() > 0) { boxes.push_back(box); scores.push_back(score); classes.push_back(class_id); }
    }
    std::vector<int> kept; cv::dnn::NMSBoxes(boxes, scores, confidence_threshold_, nms_threshold_, kept);
    FrameResult result{std::chrono::steady_clock::now(), {}};
    for (int idx : kept) result.detections.push_back({kind_for(classes[idx]), kCocoNames[classes[idx]], scores[idx], boxes[idx]});
    return result;
}

}  // namespace seatvision
