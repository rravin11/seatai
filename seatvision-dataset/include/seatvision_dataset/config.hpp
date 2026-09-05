#pragma once

#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>

namespace seatvision::dataset {

struct ModelConfig {
    std::filesystem::path engine;
    int input_width{640};
    int input_height{640};
    float confidence_threshold{0.35F};
    float nms_threshold{0.45F};
    std::vector<std::string> labels;
};

struct SemanticPolicy {
    std::unordered_set<std::string> people_labels;
    std::unordered_set<std::string> seat_labels;
    std::unordered_set<std::string> ignored_labels;
};

struct DatasetConfig {
    ModelConfig model;
    SemanticPolicy semantics;
};

DatasetConfig load_config(const std::filesystem::path& path);

}  // namespace seatvision::dataset
