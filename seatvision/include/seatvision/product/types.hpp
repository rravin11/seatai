#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace seatvision::product {

using Clock = std::chrono::steady_clock;
using Timestamp = Clock::time_point;

struct Frame {
    int camera_id{};
    std::uint64_t sequence{};
    Timestamp captured_at{};
    cv::Mat image;
};

struct Detection {
    int class_id{-1};
    std::string label;
    float score{};
    cv::Rect box;
    std::optional<cv::Mat> mask;
};

enum class EntityRole { Person, Seat, Object, Ignored, Unknown };

struct Track {
    std::uint64_t id{};
    int camera_id{};
    std::string label;
    EntityRole role{EntityRole::Unknown};
    cv::Rect box;
    float confidence{};
    std::uint64_t age{};
    std::uint64_t missed{};
    Timestamp last_seen{};
    std::optional<cv::Mat> mask;
};

enum class Occupancy { Available, Occupied, Claimed, Occluded, Unknown };

struct DynamicSeat {
    std::string id;
    int camera_id{};
    std::uint64_t source_track_id{};
    cv::Rect support_region;
    Occupancy state{Occupancy::Unknown};
    float confidence{};
    Timestamp first_seen{};
    Timestamp last_updated{};
};

struct SeatEvent {
    std::string type;
    std::string seat_id;
    int camera_id{};
    Occupancy from{Occupancy::Unknown};
    Occupancy to{Occupancy::Unknown};
    float confidence{};
    Timestamp timestamp{};
};

struct SceneSnapshot {
    int camera_id{};
    Timestamp timestamp{};
    std::vector<Track> tracks;
    std::vector<DynamicSeat> seats;
};

}  // namespace seatvision::product
