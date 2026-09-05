#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "seatvision_dataset/detector.hpp"

namespace seatvision::dataset {

struct GroundTruthBox {
    std::string label;
    cv::Rect box;
};

struct LabelMetrics {
    std::size_t ground_truth{};
    std::size_t predictions{};
    std::size_t true_positives{};
    std::size_t false_positives{};
    std::size_t false_negatives{};
    double precision{};
    double recall{};
    double f1{};
    std::optional<double> ap50;
    std::optional<double> ap50_95;
};

struct EvaluationSummary {
    std::size_t evaluated_images{};
    std::unordered_map<std::string, LabelMetrics> labels;
};

class DetectionEvaluator {
public:
    struct ImageSample {
        std::string id;
        std::vector<Detection> predictions;
        std::vector<GroundTruthBox> ground_truth;
    };

    void add_image(std::string image_id, std::vector<Detection> predictions,
                   std::vector<GroundTruthBox> ground_truth);
    [[nodiscard]] EvaluationSummary summarize() const;

private:
    std::vector<ImageSample> images_;
};

}  // namespace seatvision::dataset
