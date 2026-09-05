#pragma once

#include <unordered_map>

#include "seatvision/product/types.hpp"

namespace seatvision::product {

class SceneRenderer {
public:
    cv::Mat render(const Frame& frame, const SceneSnapshot& scene, double display_fps) const;

private:
    static cv::Scalar color_for(EntityRole role);
    static cv::Scalar color_for(Occupancy state);
    static const char* name_for(Occupancy state);
};

}  // namespace seatvision::product
