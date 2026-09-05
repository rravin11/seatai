#include "seatvision_dataset/detector.hpp"

#include <algorithm>
#include <filesystem>
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

namespace seatvision::dataset {

const char* name_for(const SemanticRole role) {
    switch (role) {
        case SemanticRole::Person: return "person";
        case SemanticRole::Seat: return "seat";
        case SemanticRole::Object: return "object";
        case SemanticRole::Ignored: return "ignored";
    }
    return "object";
}

SemanticRole classify_label(const std::string& label, const SemanticPolicy& semantics) {
    if (semantics.ignored_labels.contains(label)) return SemanticRole::Ignored;
    if (semantics.people_labels.contains(label)) return SemanticRole::Person;
    if (semantics.seat_labels.contains(label)) return SemanticRole::Seat;
    return SemanticRole::Object;
}

namespace {

class TensorRtLogger final : public nvinfer1::ILogger {
public:
    void log(const Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "TensorRT: " << message << '\n';
    }
};

TensorRtLogger& global_logger() {
    static TensorRtLogger logger;
    return logger;
}

template <typename T>
struct TensorRtDeleter {
    void operator()(T* value) const noexcept { delete value; }
};

template <typename T>
using TensorRtPtr = std::unique_ptr<T, TensorRtDeleter<T>>;

void cuda_check(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

std::size_t volume(const nvinfer1::Dims dimensions) {
    std::size_t result{1};
    for (int index = 0; index < dimensions.nbDims; ++index) {
        if (dimensions.d[index] <= 0) {
            throw std::runtime_error("Dataset detector requires a fixed-shape TensorRT engine.");
        }
        result *= static_cast<std::size_t>(dimensions.d[index]);
    }
    return result;
}

class TensorRtYoloDetector final : public Detector {
public:
    explicit TensorRtYoloDetector(ModelConfig config) : config_(std::move(config)) {
        if (!std::filesystem::is_regular_file(config_.engine)) {
            throw std::runtime_error("TensorRT engine not found: " + config_.engine.string());
        }
        std::ifstream stream(config_.engine, std::ios::binary | std::ios::ate);
        const auto bytes = stream.tellg();
        stream.seekg(0);
        std::vector<char> engine_data(static_cast<std::size_t>(bytes));
        if (!stream.read(engine_data.data(), bytes)) throw std::runtime_error("Cannot read TensorRT engine.");

        initLibNvInferPlugins(&global_logger(), "");
        runtime_.reset(nvinfer1::createInferRuntime(global_logger()));
        if (!runtime_) throw std::runtime_error("Unable to create TensorRT runtime.");
        engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
        if (!engine_) throw std::runtime_error("Unable to deserialize TensorRT engine.");
        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("Unable to create TensorRT execution context.");

        for (int index = 0; index < engine_->getNbIOTensors(); ++index) {
            const char* name = engine_->getIOTensorName(index);
            if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) input_name_ = name;
            else output_name_ = name;
        }
        if (input_name_.empty() || output_name_.empty()) {
            throw std::runtime_error("Dataset detector expects exactly one TensorRT input and output.");
        }
        const auto input_shape = engine_->getTensorShape(input_name_.c_str());
        const auto output_shape = engine_->getTensorShape(output_name_.c_str());
        if (engine_->getTensorDataType(input_name_.c_str()) != nvinfer1::DataType::kFLOAT ||
            engine_->getTensorDataType(output_name_.c_str()) != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("This adapter requires FP32 TensorRT I/O tensors.");
        }
        input_elements_ = volume(input_shape);
        output_elements_ = volume(output_shape);
        const std::size_t expected_input =
            3U * static_cast<std::size_t>(config_.input_width) * static_cast<std::size_t>(config_.input_height);
        if (input_elements_ != expected_input) {
            throw std::runtime_error("Configured input size does not match the TensorRT engine.");
        }
        if (output_shape.nbDims != 3 || output_shape.d[1] < 6 || output_shape.d[2] <= 0) {
            throw std::runtime_error("Unsupported TensorRT YOLO output shape.");
        }
        output_attributes_ = output_shape.d[1];
        output_candidates_ = output_shape.d[2];

        cuda_check(cudaStreamCreate(&stream_), "cudaStreamCreate");
        cuda_check(cudaMalloc(&device_input_, input_elements_ * sizeof(float)), "cudaMalloc input");
        cuda_check(cudaMalloc(&device_output_, output_elements_ * sizeof(float)), "cudaMalloc output");
        if (!context_->setTensorAddress(input_name_.c_str(), device_input_) ||
            !context_->setTensorAddress(output_name_.c_str(), device_output_)) {
            throw std::runtime_error("Unable to bind TensorRT input/output tensors.");
        }
    }

    ~TensorRtYoloDetector() override {
        if (device_input_) cudaFree(device_input_);
        if (device_output_) cudaFree(device_output_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    std::string name() const override { return "TensorRT/FP16"; }

    std::vector<Detection> infer(const cv::Mat& image) override {
        if (image.empty()) throw std::runtime_error("Detector received an empty image.");
        const float scale = std::min(static_cast<float>(config_.input_width) / image.cols,
                                     static_cast<float>(config_.input_height) / image.rows);
        const int resized_width = cvRound(image.cols * scale);
        const int resized_height = cvRound(image.rows * scale);
        const int pad_x = (config_.input_width - resized_width) / 2;
        const int pad_y = (config_.input_height - resized_height) / 2;
        cv::Mat padded(config_.input_height, config_.input_width, CV_8UC3, cv::Scalar(114, 114, 114));
        cv::resize(image, padded(cv::Rect(pad_x, pad_y, resized_width, resized_height)),
                   {resized_width, resized_height});

        std::vector<float> input(input_elements_);
        const int plane = config_.input_width * config_.input_height;
        for (int y = 0; y < config_.input_height; ++y) {
            for (int x = 0; x < config_.input_width; ++x) {
                const cv::Vec3b pixel = padded.at<cv::Vec3b>(y, x);
                input[x + y * config_.input_width] = pixel[2] / 255.0F;
                input[plane + x + y * config_.input_width] = pixel[1] / 255.0F;
                input[2 * plane + x + y * config_.input_width] = pixel[0] / 255.0F;
            }
        }
        std::vector<float> output(output_elements_);
        cuda_check(cudaMemcpyAsync(device_input_, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice,
                                   stream_),
                   "copy input");
        if (!context_->enqueueV3(stream_)) throw std::runtime_error("TensorRT enqueueV3 failed.");
        cuda_check(cudaMemcpyAsync(output.data(), device_output_, output.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                   stream_),
                   "copy output");
        cuda_check(cudaStreamSynchronize(stream_), "synchronize inference");
        return postprocess(output, image.size(), scale, pad_x, pad_y);
    }

private:
    std::vector<Detection> postprocess(const std::vector<float>& output, const cv::Size& original, const float scale,
                                       const int pad_x, const int pad_y) const {
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> class_ids;
        for (int candidate = 0; candidate < output_candidates_; ++candidate) {
            int class_id{};
            float score{};
            for (int label_index = 0; label_index < output_attributes_ - 4; ++label_index) {
                const float candidate_score = output[(4 + label_index) * output_candidates_ + candidate];
                if (candidate_score > score) {
                    score = candidate_score;
                    class_id = label_index;
                }
            }
            if (score < config_.confidence_threshold || class_id >= static_cast<int>(config_.labels.size())) continue;
            const float cx = output[candidate];
            const float cy = output[output_candidates_ + candidate];
            const float width = output[2 * output_candidates_ + candidate];
            const float height = output[3 * output_candidates_ + candidate];
            cv::Rect box(cvRound((cx - width * 0.5F - pad_x) / scale),
                         cvRound((cy - height * 0.5F - pad_y) / scale), cvRound(width / scale),
                         cvRound(height / scale));
            box &= {0, 0, original.width, original.height};
            if (box.area() > 0) {
                boxes.push_back(box);
                scores.push_back(score);
                class_ids.push_back(class_id);
            }
        }
        std::vector<Detection> detections;
        for (int class_id = 0; class_id < static_cast<int>(config_.labels.size()); ++class_id) {
            std::vector<cv::Rect> grouped_boxes;
            std::vector<float> grouped_scores;
            std::vector<int> original_indices;
            for (std::size_t index = 0; index < class_ids.size(); ++index) {
                if (class_ids[index] != class_id) continue;
                grouped_boxes.push_back(boxes[index]);
                grouped_scores.push_back(scores[index]);
                original_indices.push_back(static_cast<int>(index));
            }
            std::vector<int> kept;
            cv::dnn::NMSBoxes(grouped_boxes, grouped_scores, config_.confidence_threshold, config_.nms_threshold, kept);
            for (const int index : kept) {
                const int original_index = original_indices[index];
                detections.push_back(
                    {class_id, config_.labels[class_id], scores[original_index], boxes[original_index]});
            }
        }
        return detections;
    }

    ModelConfig config_;
    TensorRtPtr<nvinfer1::IRuntime> runtime_{nullptr};
    TensorRtPtr<nvinfer1::ICudaEngine> engine_{nullptr};
    TensorRtPtr<nvinfer1::IExecutionContext> context_{nullptr};
    std::string input_name_;
    std::string output_name_;
    std::size_t input_elements_{};
    std::size_t output_elements_{};
    int output_attributes_{};
    int output_candidates_{};
    cudaStream_t stream_{};
    void* device_input_{};
    void* device_output_{};
};

}  // namespace

std::unique_ptr<Detector> create_detector(const ModelConfig& config) {
    return std::make_unique<TensorRtYoloDetector>(config);
}

}  // namespace seatvision::dataset
