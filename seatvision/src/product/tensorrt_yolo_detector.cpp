#include "seatvision/product/detector.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace seatvision::product {
namespace {
class TensorRtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "TensorRT: " << message << '\n';
    }
};

TensorRtLogger& global_logger() {
    static TensorRtLogger logger;
    return logger;
}

template <typename T> struct TensorRtDeleter { void operator()(T* value) const noexcept { delete value; } };
template <typename T> using TensorRtPtr = std::unique_ptr<T, TensorRtDeleter<T>>;

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

std::size_t volume(nvinfer1::Dims dims) {
    std::size_t result = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) throw std::runtime_error("Dynamic TensorRT dimensions require a profile; this runtime uses fixed-size engines.");
        result *= static_cast<std::size_t>(dims.d[i]);
    }
    return result;
}

class TensorRtYoloDetector final : public Detector {
public:
    explicit TensorRtYoloDetector(DetectorConfig config) : config_(std::move(config)) {
        std::ifstream stream(config_.engine, std::ios::binary | std::ios::ate);
        if (!stream) throw std::runtime_error("TensorRT engine not found: " + config_.engine.string());
        const auto size = stream.tellg(); stream.seekg(0);
        std::vector<char> data(static_cast<std::size_t>(size)); stream.read(data.data(), size);
        initLibNvInferPlugins(&global_logger(), "");
        runtime_.reset(nvinfer1::createInferRuntime(global_logger()));
        if (!runtime_) throw std::runtime_error("Unable to create TensorRT runtime.");
        engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
        if (!engine_) throw std::runtime_error("Unable to deserialize TensorRT engine.");
        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("Unable to create TensorRT execution context.");
        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* name = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) input_name_ = name;
            else output_name_ = name;
        }
        if (input_name_.empty() || output_name_.empty()) throw std::runtime_error("Engine needs one input and one output tensor.");
        const auto input_shape = engine_->getTensorShape(input_name_.c_str());
        const auto output_shape = engine_->getTensorShape(output_name_.c_str());
        if (engine_->getTensorDataType(input_name_.c_str()) != nvinfer1::DataType::kFLOAT ||
            engine_->getTensorDataType(output_name_.c_str()) != nvinfer1::DataType::kFLOAT)
            throw std::runtime_error("This starter TensorRT adapter expects FP32 I/O tensors; build the engine without --inputIOFormats fp16.");
        input_elements_ = volume(input_shape); output_elements_ = volume(output_shape);
        if (output_shape.nbDims != 3 || output_shape.d[1] < 6) throw std::runtime_error("Unsupported TensorRT YOLO output tensor shape.");
        output_attributes_ = output_shape.d[1]; output_candidates_ = output_shape.d[2];
        cuda_check(cudaStreamCreate(&stream_), "cudaStreamCreate");
        cuda_check(cudaMalloc(&device_input_, input_elements_ * sizeof(float)), "cudaMalloc input");
        cuda_check(cudaMalloc(&device_output_, output_elements_ * sizeof(float)), "cudaMalloc output");
        if (!context_->setTensorAddress(input_name_.c_str(), device_input_) || !context_->setTensorAddress(output_name_.c_str(), device_output_))
            throw std::runtime_error("Unable to bind TensorRT tensors.");
    }

    ~TensorRtYoloDetector() override {
        if (device_input_) cudaFree(device_input_);
        if (device_output_) cudaFree(device_output_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    std::string name() const override { return "TensorRT/FP16"; }

    std::vector<Detection> infer(const cv::Mat& frame) override {
        const float scale = std::min(static_cast<float>(config_.input_width) / frame.cols,
                                     static_cast<float>(config_.input_height) / frame.rows);
        const int resized_width = cvRound(frame.cols * scale), resized_height = cvRound(frame.rows * scale);
        const int pad_x = (config_.input_width - resized_width) / 2, pad_y = (config_.input_height - resized_height) / 2;
        cv::Mat padded(config_.input_height, config_.input_width, CV_8UC3, cv::Scalar(114, 114, 114));
        cv::resize(frame, padded(cv::Rect(pad_x, pad_y, resized_width, resized_height)), cv::Size(resized_width, resized_height));
        std::vector<float> input(input_elements_);
        const int plane = config_.input_width * config_.input_height;
        for (int y = 0; y < config_.input_height; ++y) for (int x = 0; x < config_.input_width; ++x) {
            const cv::Vec3b pixel = padded.at<cv::Vec3b>(y, x);
            input[x + y * config_.input_width] = pixel[2] / 255.0F;
            input[plane + x + y * config_.input_width] = pixel[1] / 255.0F;
            input[2 * plane + x + y * config_.input_width] = pixel[0] / 255.0F;
        }
        std::vector<float> output(output_elements_);
        cuda_check(cudaMemcpyAsync(device_input_, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, stream_), "copy input");
        if (!context_->enqueueV3(stream_)) throw std::runtime_error("TensorRT enqueueV3 failed.");
        cuda_check(cudaMemcpyAsync(output.data(), device_output_, output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_), "copy output");
        cuda_check(cudaStreamSynchronize(stream_), "synchronize inference");
        return postprocess(output, frame.size(), scale, pad_x, pad_y);
    }

private:
    std::vector<Detection> postprocess(const std::vector<float>& output, const cv::Size& original, float scale, int pad_x, int pad_y) const {
        std::vector<cv::Rect> boxes; std::vector<float> scores; std::vector<int> class_ids;
        for (int candidate = 0; candidate < output_candidates_; ++candidate) {
            const float* values = output.data() + candidate;
            int class_id{}; float score{};
            for (int klass = 0; klass < output_attributes_ - 4; ++klass) {
                const float candidate_score = output[(4 + klass) * output_candidates_ + candidate];
                if (candidate_score > score) { score = candidate_score; class_id = klass; }
            }
            if (score < config_.confidence_threshold || class_id >= static_cast<int>(config_.labels.size())) continue;
            const float cx = values[0 * output_candidates_];
            const float cy = values[1 * output_candidates_];
            const float width = values[2 * output_candidates_];
            const float height = values[3 * output_candidates_];
            cv::Rect box(cvRound((cx - width * 0.5F - pad_x) / scale), cvRound((cy - height * 0.5F - pad_y) / scale),
                         cvRound(width / scale), cvRound(height / scale));
            box &= cv::Rect(0, 0, original.width, original.height);
            if (box.area() > 0) { boxes.push_back(box); scores.push_back(score); class_ids.push_back(class_id); }
        }
        std::vector<Detection> detections;
        for (int class_id = 0; class_id < static_cast<int>(config_.labels.size()); ++class_id) {
            std::vector<cv::Rect> group_boxes; std::vector<float> group_scores; std::vector<int> original_indices;
            for (std::size_t i = 0; i < class_ids.size(); ++i) if (class_ids[i] == class_id) {
                group_boxes.push_back(boxes[i]); group_scores.push_back(scores[i]); original_indices.push_back(static_cast<int>(i));
            }
            std::vector<int> kept; cv::dnn::NMSBoxes(group_boxes, group_scores, config_.confidence_threshold, config_.nms_threshold, kept);
            for (int index : kept) { const int original_index = original_indices[index]; detections.push_back({class_id, config_.labels[class_id], scores[original_index], boxes[original_index], std::nullopt}); }
        }
        return detections;
    }

    DetectorConfig config_;
    TensorRtPtr<nvinfer1::IRuntime> runtime_{nullptr}; TensorRtPtr<nvinfer1::ICudaEngine> engine_{nullptr}; TensorRtPtr<nvinfer1::IExecutionContext> context_{nullptr};
    std::string input_name_, output_name_; std::size_t input_elements_{}, output_elements_{}; int output_attributes_{}, output_candidates_{};
    cudaStream_t stream_{}; void* device_input_{}; void* device_output_{};
};
}  // namespace

std::unique_ptr<Detector> create_tensorrt_yolo_detector(const DetectorConfig& config) { return std::make_unique<TensorRtYoloDetector>(config); }

}  // namespace seatvision::product
