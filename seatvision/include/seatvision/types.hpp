#pragma once

#include <chrono>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace seatvision {

enum class ObjectKind { Person, Chair, Backpack, Bottle, Other };

struct Detection {
    ObjectKind kind{ObjectKind::Other};
    std::string label;
    float confidence{};
    cv::Rect box;
};

struct FrameResult {
    std::chrono::steady_clock::time_point captured_at;
    std::vector<Detection> detections;
};

enum class SeatState { Available, Occupied, Claimed, Uncertain };

struct SeatZone {
    std::string id;
    int camera_id{};
    std::vector<cv::Point2f> normalized_polygon;
};

struct SeatEstimate {
    std::string id;
    SeatState state{SeatState::Uncertain};
    float confidence{};
};

}  // namespace seatvision
