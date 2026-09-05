#include "seatvision_dataset/evaluation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <tuple>
#include <utility>

namespace seatvision::dataset {
namespace {

float iou(const cv::Rect& first, const cv::Rect& second) {
    const int intersection = (first & second).area();
    const int combined = first.area() + second.area() - intersection;
    return combined > 0 ? static_cast<float>(intersection) / static_cast<float>(combined) : 0.0F;
}

struct Counts {
    std::size_t ground_truth{};
    std::size_t predictions{};
    std::size_t true_positives{};
    std::size_t false_positives{};
    std::size_t false_negatives{};
    std::optional<double> average_precision;
};

struct IndexedPrediction {
    std::size_t image_index{};
    const Detection* detection{};
};

Counts evaluate_label(const std::vector<DetectionEvaluator::ImageSample>& images, const std::string& label,
                      const float iou_threshold) {
    Counts result;
    std::vector<std::vector<bool>> matched(images.size());
    std::vector<IndexedPrediction> predictions;
    for (std::size_t image_index = 0; image_index < images.size(); ++image_index) {
        const auto& image = images[image_index];
        matched[image_index].resize(image.ground_truth.size());
        for (const auto& target : image.ground_truth) if (target.label == label) ++result.ground_truth;
        for (const auto& prediction : image.predictions) {
            if (prediction.label == label) predictions.push_back({image_index, &prediction});
        }
    }
    std::sort(predictions.begin(), predictions.end(), [](const auto& left, const auto& right) {
        if (left.detection->score != right.detection->score) return left.detection->score > right.detection->score;
        if (left.image_index != right.image_index) return left.image_index < right.image_index;
        const auto& a = left.detection->box;
        const auto& b = right.detection->box;
        return std::tie(a.x, a.y, a.width, a.height) < std::tie(b.x, b.y, b.width, b.height);
    });
    result.predictions = predictions.size();

    std::vector<double> precision_points;
    std::vector<double> recall_points;
    for (const auto& prediction : predictions) {
        const auto& image = images[prediction.image_index];
        int best_target{-1};
        float best_iou{iou_threshold};
        for (std::size_t target_index = 0; target_index < image.ground_truth.size(); ++target_index) {
            const auto& target = image.ground_truth[target_index];
            if (matched[prediction.image_index][target_index] || target.label != label) continue;
            const float candidate_iou = iou(prediction.detection->box, target.box);
            if (candidate_iou >= best_iou) {
                best_iou = candidate_iou;
                best_target = static_cast<int>(target_index);
            }
        }
        if (best_target >= 0) {
            matched[prediction.image_index][static_cast<std::size_t>(best_target)] = true;
            ++result.true_positives;
        } else {
            ++result.false_positives;
        }
        const double prediction_count = static_cast<double>(result.true_positives + result.false_positives);
        precision_points.push_back(prediction_count > 0.0 ? result.true_positives / prediction_count : 0.0);
        recall_points.push_back(result.ground_truth > 0 ? static_cast<double>(result.true_positives) / result.ground_truth
                                                         : 0.0);
    }
    result.false_negatives = result.ground_truth - result.true_positives;
    if (result.ground_truth == 0) return result;

    for (std::size_t index = precision_points.size(); index > 1; --index) {
        precision_points[index - 2] = std::max(precision_points[index - 2], precision_points[index - 1]);
    }
    double prior_recall{};
    double average_precision{};
    for (std::size_t index = 0; index < recall_points.size(); ++index) {
        average_precision += std::max(0.0, recall_points[index] - prior_recall) * precision_points[index];
        prior_recall = recall_points[index];
    }
    result.average_precision = average_precision;
    return result;
}

}  // namespace

void DetectionEvaluator::add_image(std::string image_id, std::vector<Detection> predictions,
                                   std::vector<GroundTruthBox> ground_truth) {
    images_.push_back({std::move(image_id), std::move(predictions), std::move(ground_truth)});
}

EvaluationSummary DetectionEvaluator::summarize() const {
    EvaluationSummary summary;
    summary.evaluated_images = images_.size();
    std::set<std::string> labels;
    for (const auto& image : images_) {
        for (const auto& prediction : image.predictions) labels.insert(prediction.label);
        for (const auto& target : image.ground_truth) labels.insert(target.label);
    }
    constexpr std::array<float, 10> iou_thresholds{0.50F, 0.55F, 0.60F, 0.65F, 0.70F,
                                                     0.75F, 0.80F, 0.85F, 0.90F, 0.95F};
    for (const auto& label : labels) {
        const Counts at_fifty = evaluate_label(images_, label, iou_thresholds.front());
        LabelMetrics metrics;
        metrics.ground_truth = at_fifty.ground_truth;
        metrics.predictions = at_fifty.predictions;
        metrics.true_positives = at_fifty.true_positives;
        metrics.false_positives = at_fifty.false_positives;
        metrics.false_negatives = at_fifty.false_negatives;
        const double precision_denominator = static_cast<double>(metrics.true_positives + metrics.false_positives);
        const double recall_denominator = static_cast<double>(metrics.true_positives + metrics.false_negatives);
        metrics.precision = precision_denominator > 0.0 ? metrics.true_positives / precision_denominator : 0.0;
        metrics.recall = recall_denominator > 0.0 ? metrics.true_positives / recall_denominator : 0.0;
        metrics.f1 = metrics.precision + metrics.recall > 0.0
                         ? 2.0 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
                         : 0.0;
        metrics.ap50 = at_fifty.average_precision;
        if (at_fifty.average_precision.has_value()) {
            double total{};
            for (const float threshold : iou_thresholds) {
                total += *evaluate_label(images_, label, threshold).average_precision;
            }
            metrics.ap50_95 = total / static_cast<double>(iou_thresholds.size());
        }
        summary.labels.emplace(label, std::move(metrics));
    }
    return summary;
}

}  // namespace seatvision::dataset
