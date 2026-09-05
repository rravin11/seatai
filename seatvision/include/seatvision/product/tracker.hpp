#pragma once

#include <unordered_map>
#include <vector>

#include "seatvision/product/config.hpp"
#include "seatvision/product/types.hpp"

namespace seatvision::product {

class MultiObjectTracker {
public:
    explicit MultiObjectTracker(ReasonerConfig config);
    std::vector<Track> update(int camera_id, Timestamp timestamp, const std::vector<Detection>& detections,
                              const SemanticPolicy& semantics);

private:
    EntityRole classify(const std::string& label, const SemanticPolicy& semantics) const;
    ReasonerConfig config_;
    std::uint64_t next_id_{1};
    std::unordered_map<int, std::vector<Track>> tracks_by_camera_;
};

}  // namespace seatvision::product
