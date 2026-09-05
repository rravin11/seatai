#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "seatvision_dataset/config.hpp"

namespace seatvision::dataset {

struct Detection {
    int class_id{-1};
    std::string label;
    float score{};
    cv::Rect box;
};

enum class SemanticRole { Person, Seat, Object, Ignored };

const char* name_for(SemanticRole role);
SemanticRole classify_label(const std::string& label, const SemanticPolicy& semantics);

class Detector {
public:
    virtual ~Detector() = default;
    virtual std::vector<Detection> infer(const cv::Mat& image) = 0;
    [[nodiscard]] virtual std::string name() const = 0;
};

std::unique_ptr<Detector> create_detector(const ModelConfig& config);

}  // namespace seatvision::dataset
