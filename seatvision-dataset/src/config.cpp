#include "seatvision_dataset/config.hpp"

#include <algorithm>
#include <stdexcept>

#include <opencv2/core.hpp>

namespace seatvision::dataset {
namespace {

std::vector<std::string> strings(const cv::FileNode& node) {
    std::vector<std::string> values;
    if (node.empty()) return values;
    for (const auto& value : node) values.push_back(static_cast<std::string>(value));
    return values;
}

std::unordered_set<std::string> string_set(const cv::FileNode& node) {
    const auto values = strings(node);
    return {values.begin(), values.end()};
}

std::filesystem::path resolve_from_config(const std::filesystem::path& config_path, const std::string& value) {
    const std::filesystem::path supplied{value};
    return supplied.is_absolute() ? supplied : (config_path.parent_path() / supplied).lexically_normal();
}

}  // namespace

DatasetConfig load_config(const std::filesystem::path& path) {
    cv::FileStorage storage(path.string(), cv::FileStorage::READ);
    if (!storage.isOpened()) throw std::runtime_error("Cannot open dataset configuration: " + path.string());
    const cv::FileNode model = storage["model"];
    if (model.empty()) throw std::runtime_error("Dataset configuration requires a model section.");

    DatasetConfig config;
    std::string engine;
    model["engine"] >> engine;
    if (engine.empty()) throw std::runtime_error("Dataset configuration requires model.engine.");
    config.model.engine = resolve_from_config(path, engine);
    model["input_width"] >> config.model.input_width;
    model["input_height"] >> config.model.input_height;
    model["confidence_threshold"] >> config.model.confidence_threshold;
    model["nms_threshold"] >> config.model.nms_threshold;
    config.model.labels = strings(model["labels"]);
    if (config.model.labels.empty()) throw std::runtime_error("Dataset configuration requires model.labels.");

    const cv::FileNode semantics = storage["semantics"];
    if (!semantics.empty()) {
        config.semantics.people_labels = string_set(semantics["people_labels"]);
        config.semantics.seat_labels = string_set(semantics["seat_labels"]);
        config.semantics.ignored_labels = string_set(semantics["ignored_labels"]);
    }
    config.model.input_width = std::max(1, config.model.input_width);
    config.model.input_height = std::max(1, config.model.input_height);
    config.model.confidence_threshold = std::clamp(config.model.confidence_threshold, 0.0F, 1.0F);
    config.model.nms_threshold = std::clamp(config.model.nms_threshold, 0.0F, 1.0F);
    return config;
}

}  // namespace seatvision::dataset
