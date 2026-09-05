#pragma once

#include <condition_variable>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "seatvision/types.hpp"

namespace seatvision {

class AsyncYoloDetector {
public:
    AsyncYoloDetector(std::string model_path, float confidence_threshold, float nms_threshold);
    ~AsyncYoloDetector();

    AsyncYoloDetector(const AsyncYoloDetector&) = delete;
    AsyncYoloDetector& operator=(const AsyncYoloDetector&) = delete;

    bool ready() const;
    void submit(const cv::Mat& frame);
    std::optional<FrameResult> latest();

private:
    void worker();
    FrameResult infer(const cv::Mat& frame) const;

    std::string model_path_;
    float confidence_threshold_;
    float nms_threshold_;
    mutable std::mutex mutex_;
    std::condition_variable work_ready_;
    std::optional<cv::Mat> pending_frame_;
    std::optional<FrameResult> latest_result_;
    bool stop_{false};
    bool ready_{false};
    std::thread worker_;
};

}  // namespace seatvision
