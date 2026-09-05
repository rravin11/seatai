#pragma once

#include <unordered_map>
#include <vector>

#include "seatvision/product/config.hpp"
#include "seatvision/product/types.hpp"

namespace seatvision::product {

class DynamicSeatReasoner {
public:
    explicit DynamicSeatReasoner(ReasonerConfig config);
    SceneSnapshot update(int camera_id, Timestamp timestamp, const std::vector<Track>& tracks);
    std::vector<SeatEvent> take_events();

private:
    struct SeatMemory {
        DynamicSeat seat;
        Occupancy candidate{Occupancy::Unknown};
        Timestamp candidate_since{};
    };

    static float intersection_over_union(const cv::Rect& a, const cv::Rect& b);
    static float overlap_ratio(const cv::Rect& container, const cv::Rect& candidate);
    static Occupancy choose_evidence(const DynamicSeat& seat, const std::vector<Track>& tracks, float& confidence);
    std::chrono::milliseconds confirmation_for(Occupancy state) const;

    ReasonerConfig config_;
    std::unordered_map<std::string, SeatMemory> seats_;
    std::vector<SeatEvent> pending_events_;
};

}  // namespace seatvision::product
