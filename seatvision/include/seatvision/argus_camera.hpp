#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <mutex>

typedef struct _GstElement GstElement;

namespace seatvision {

class ArgusCamera {
public:
    ArgusCamera(int sensor_id, int width, int height, int fps, bool rotate_180 = false);
    ~ArgusCamera();
    bool open();
    bool read(cv::Mat& frame);
    [[nodiscard]] int sensor_id() const { return sensor_id_; }

private:
    int sensor_id_;
    int width_;
    int height_;
    int fps_;
    bool rotate_180_;
    GstElement* pipeline_{nullptr};
    GstElement* appsink_{nullptr};
};

}  // namespace seatvision
