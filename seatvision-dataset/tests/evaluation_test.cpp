#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>

#include "seatvision_dataset/evaluation.hpp"

namespace {

void require(const bool condition, const char* message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(1);
    }
}

seatvision::dataset::Detection detection(const std::string& label, const float score, const cv::Rect box) {
    return {-1, label, score, box};
}

}  // namespace

int main() {
    using namespace seatvision::dataset;
    DetectionEvaluator evaluator;
    evaluator.add_image(
        "scene-1.jpg",
        {detection("person", 0.95F, {10, 10, 20, 20}), detection("person", 0.20F, {80, 80, 10, 10}),
         detection("bottle", 0.70F, {40, 40, 10, 10})},
        {{"person", {10, 10, 20, 20}}, {"chair", {50, 50, 20, 20}}});
    const auto summary = evaluator.summarize();

    const auto person = summary.labels.at("person");
    require(person.ground_truth == 1 && person.predictions == 2, "person counts");
    require(person.true_positives == 1 && person.false_positives == 1 && person.false_negatives == 0,
            "person TP/FP/FN");
    require(std::abs(person.precision - 0.5) < 0.0001, "person precision");
    require(std::abs(person.recall - 1.0) < 0.0001, "person recall");
    require(person.ap50.has_value() && std::abs(*person.ap50 - 1.0) < 0.0001, "person AP50");
    require(person.ap50_95.has_value() && std::abs(*person.ap50_95 - 1.0) < 0.0001, "person AP50:95");

    const auto chair = summary.labels.at("chair");
    require(chair.ground_truth == 1 && chair.predictions == 0 && chair.false_negatives == 1, "chair missed");
    require(chair.ap50.has_value() && std::abs(*chair.ap50) < 0.0001, "chair AP50");

    const auto bottle = summary.labels.at("bottle");
    require(bottle.ground_truth == 0 && bottle.predictions == 1, "bottle counts");
    require(!bottle.ap50.has_value() && !bottle.ap50_95.has_value(), "no-ground-truth AP is null");
    return 0;
}
