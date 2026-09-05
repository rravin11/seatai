#pragma once

#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>

namespace seatvision::product {

struct CameraConfig {
    int sensor_id{};
    int width{1280};
    int height{720};
    int fps{30};
    // Applied in the NVMM path before inference, tracking, events, and display.
    bool rotate_180{false};
};

struct DetectorConfig {
    std::string backend{"tensorrt"};
    std::filesystem::path onnx_model{"models/yolov8n.onnx"};
    std::filesystem::path engine{"models/yolov8n_fp16_orin.engine"};
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

struct ReasonerConfig {
    std::uint64_t minimum_seat_track_age{12};
    std::uint64_t track_ttl_frames{45};
    float association_iou{0.25F};
    std::chrono::milliseconds occupied_confirmation{700};
    std::chrono::milliseconds claimed_confirmation{1200};
    std::chrono::milliseconds available_confirmation{2500};
};

struct RuntimeConfig {
    std::vector<CameraConfig> cameras;
    DetectorConfig detector;
    SemanticPolicy semantics;
    ReasonerConfig reasoner;
    bool show_window{true};
    // Display-only dimensions. Inference always uses the camera's native frame.
    int preview_tile_width{800};
    int preview_tile_height{450};
    int inference_interval{4};
    std::filesystem::path event_log{"runtime/seat-events.jsonl"};
};

RuntimeConfig load_config(const std::filesystem::path& path);

}  // namespace seatvision::product
