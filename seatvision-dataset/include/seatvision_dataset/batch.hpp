#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>

#include "seatvision_dataset/config.hpp"

namespace seatvision::dataset {

struct BatchOptions {
    std::filesystem::path input;
    std::filesystem::path output;
    std::optional<std::filesystem::path> coco_annotations;
    std::unordered_map<std::string, std::string> label_aliases;
    bool recursive{true};
    bool save_annotated{true};
    std::size_t limit{};          // Zero means no limit.
    std::size_t gallery_limit{120};
};

// Runs local, static detector validation. It never produces temporal
// availability or occupancy decisions from individual photographs.
int run_batch(const DatasetConfig& config, const BatchOptions& options);

}  // namespace seatvision::dataset
