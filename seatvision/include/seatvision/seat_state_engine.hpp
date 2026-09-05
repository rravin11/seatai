#pragma once

#include <chrono>
#include <unordered_map>
#include <vector>

#include "seatvision/types.hpp"

namespace seatvision {

class SeatStateEngine {
public:
    explicit SeatStateEngine(std::vector<SeatZone> zones);
    std::vector<SeatEstimate> update(int camera_id, const cv::Size& frame_size,
                                     const std::vector<Detection>& detections);
    void draw(cv::Mat& frame, int camera_id) const;

private:
    struct Memory {
        SeatState stable{SeatState::Available};
        SeatState candidate{SeatState::Available};
        std::chrono::steady_clock::time_point candidate_since{};
        float confidence{};
    };

    static const char* state_name(SeatState state);
    static cv::Scalar state_color(SeatState state);
    static bool overlaps(const std::vector<cv::Point>& zone, const cv::Rect& box);
    SeatState observe(const std::vector<cv::Point>& zone, const std::vector<Detection>& detections,
                      float& confidence) const;

    std::vector<SeatZone> zones_;
    std::unordered_map<std::string, Memory> memory_;
};

}  // namespace seatvision
