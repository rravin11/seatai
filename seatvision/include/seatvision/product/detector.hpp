#pragma once

#include <memory>
#include <vector>

#include "seatvision/product/config.hpp"
#include "seatvision/product/types.hpp"

namespace seatvision::product {

class Detector {
public:
    virtual ~Detector() = default;
    virtual std::vector<Detection> infer(const cv::Mat& frame) = 0;
    [[nodiscard]] virtual std::string name() const = 0;
};

std::unique_ptr<Detector> create_detector(const DetectorConfig& config);

}  // namespace seatvision::product
