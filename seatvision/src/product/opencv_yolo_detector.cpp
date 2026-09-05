#include "seatvision/product/detector.hpp"

#include <algorithm>
#include <filesystem>
#include <stdexcept>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace seatvision::product {
namespace {
class OpenCvYoloDetector final : public Detector {
public:
    explicit OpenCvYoloDetector(DetectorConfig config) : config_(std::move(config)) {
        if (!std::filesystem::is_regular_file(config_.onnx_model))
            throw std::runtime_error("ONNX model does not exist: " + config_.onnx_model.string());
        network_ = cv::dnn::readNetFromONNX(config_.onnx_model.string());
        network_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        network_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    std::string name() const override { return "OpenCV-DNN/ONNX"; }

    std::vector<Detection> infer(const cv::Mat& frame) override {
        if (frame.empty()) return {};
        const float scale = std::min(static_cast<float>(config_.input_width) / frame.cols,
                                     static_cast<float>(config_.input_height) / frame.rows);
        const int resized_width = cvRound(frame.cols * scale);
        const int resized_height = cvRound(frame.rows * scale);
        const int pad_x = (config_.input_width - resized_width) / 2;
        const int pad_y = (config_.input_height - resized_height) / 2;
        cv::Mat input(config_.input_height, config_.input_width, CV_8UC3, cv::Scalar(114, 114, 114));
        cv::resize(frame, input(cv::Rect(pad_x, pad_y, resized_width, resized_height)), cv::Size(resized_width, resized_height));
        network_.setInput(cv::dnn::blobFromImage(input, 1.0 / 255.0, {}, {}, true, false));
        cv::Mat output = network_.forward();
        if (output.dims != 3) throw std::runtime_error("Unsupported YOLO output rank: " + std::to_string(output.dims));
        const int attributes = output.size[1];
        const int candidates = output.size[2];
        if (attributes < 6) throw std::runtime_error("Unsupported YOLO attributes: " + std::to_string(attributes));
        cv::Mat rows;
        cv::transpose(cv::Mat(attributes, candidates, CV_32F, output.ptr<float>()), rows);
        std::vector<cv::Rect> boxes; std::vector<float> scores; std::vector<int> class_ids;
        for (int row = 0; row < rows.rows; ++row) {
            const float* values = rows.ptr<float>(row);
            const auto start = values + 4;
            const auto end = values + attributes;
            const int class_id = static_cast<int>(std::max_element(start, end) - start);
            const float score = values[4 + class_id];
            if (score < config_.confidence_threshold) continue;
            cv::Rect box(cvRound((values[0] - values[2] * 0.5F - pad_x) / scale),
                         cvRound((values[1] - values[3] * 0.5F - pad_y) / scale),
                         cvRound(values[2] / scale), cvRound(values[3] / scale));
            box &= cv::Rect(0, 0, frame.cols, frame.rows);
            if (box.area() > 0) { boxes.push_back(box); scores.push_back(score); class_ids.push_back(class_id); }
        }
        // NMS is intentionally per class so a person sitting in a chair does not suppress the chair.
        std::vector<Detection> detections;
        for (int class_id = 0; class_id < static_cast<int>(config_.labels.size()); ++class_id) {
            std::vector<cv::Rect> class_boxes; std::vector<float> class_scores; std::vector<int> original;
            for (std::size_t i = 0; i < class_ids.size(); ++i) if (class_ids[i] == class_id) {
                original.push_back(static_cast<int>(i)); class_boxes.push_back(boxes[i]); class_scores.push_back(scores[i]);
            }
            std::vector<int> kept; cv::dnn::NMSBoxes(class_boxes, class_scores, config_.confidence_threshold, config_.nms_threshold, kept);
            for (const int kept_index : kept) {
                const int index = original[kept_index];
                detections.push_back({class_id, config_.labels[class_id], scores[index], boxes[index], std::nullopt});
            }
        }
        return detections;
    }

private:
    DetectorConfig config_;
    cv::dnn::Net network_;
};
}  // namespace

std::unique_ptr<Detector> create_opencv_yolo_detector(const DetectorConfig& config) {
    return std::make_unique<OpenCvYoloDetector>(config);
}

}  // namespace seatvision::product
