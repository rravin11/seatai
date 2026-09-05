#include "seatvision/product/config.hpp"

#include <algorithm>
#include <stdexcept>

#include <opencv2/core.hpp>

namespace seatvision::product {
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
}  // namespace

RuntimeConfig load_config(const std::filesystem::path& path) {
    cv::FileStorage fs(path.string(), cv::FileStorage::READ);
    if (!fs.isOpened()) throw std::runtime_error("Cannot open configuration: " + path.string());
    RuntimeConfig config;
    for (const auto& camera : fs["cameras"]) {
        CameraConfig entry;
        camera["sensor_id"] >> entry.sensor_id;
        camera["width"] >> entry.width;
        camera["height"] >> entry.height;
        camera["fps"] >> entry.fps;
        const auto rotate_180 = camera["rotate_180"];
        if (!rotate_180.empty()) {
            int enabled{};
            rotate_180 >> enabled;
            entry.rotate_180 = enabled != 0;
        }
        config.cameras.push_back(entry);
    }
    if (config.cameras.empty()) throw std::runtime_error("Configuration needs at least one camera.");
    const auto detector = fs["detector"];
    if (!detector.empty()) {
        detector["backend"] >> config.detector.backend;
        std::string text;
        detector["onnx_model"] >> text; if (!text.empty()) config.detector.onnx_model = text;
        text.clear(); detector["engine"] >> text; if (!text.empty()) config.detector.engine = text;
        detector["input_width"] >> config.detector.input_width;
        detector["input_height"] >> config.detector.input_height;
        detector["confidence_threshold"] >> config.detector.confidence_threshold;
        detector["nms_threshold"] >> config.detector.nms_threshold;
        config.detector.labels = strings(detector["labels"]);
    }
    const auto semantic = fs["semantics"];
    if (!semantic.empty()) {
        config.semantics.people_labels = string_set(semantic["people_labels"]);
        config.semantics.seat_labels = string_set(semantic["seat_labels"]);
        config.semantics.ignored_labels = string_set(semantic["ignored_labels"]);
    }
    const auto reasoner = fs["reasoner"];
    if (!reasoner.empty()) {
        int minimum_seat_track_age = static_cast<int>(config.reasoner.minimum_seat_track_age);
        int track_ttl_frames = static_cast<int>(config.reasoner.track_ttl_frames);
        reasoner["minimum_seat_track_age"] >> minimum_seat_track_age;
        reasoner["track_ttl_frames"] >> track_ttl_frames;
        config.reasoner.minimum_seat_track_age = static_cast<std::uint64_t>(std::max(1, minimum_seat_track_age));
        config.reasoner.track_ttl_frames = static_cast<std::uint64_t>(std::max(1, track_ttl_frames));
        reasoner["association_iou"] >> config.reasoner.association_iou;
        int occupied_ms = static_cast<int>(config.reasoner.occupied_confirmation.count());
        int claimed_ms = static_cast<int>(config.reasoner.claimed_confirmation.count());
        int available_ms = static_cast<int>(config.reasoner.available_confirmation.count());
        reasoner["occupied_confirmation_ms"] >> occupied_ms;
        reasoner["claimed_confirmation_ms"] >> claimed_ms;
        reasoner["available_confirmation_ms"] >> available_ms;
        config.reasoner.occupied_confirmation = std::chrono::milliseconds(occupied_ms);
        config.reasoner.claimed_confirmation = std::chrono::milliseconds(claimed_ms);
        config.reasoner.available_confirmation = std::chrono::milliseconds(available_ms);
    }
    const auto runtime = fs["runtime"];
    if (!runtime.empty()) {
        runtime["show_window"] >> config.show_window;
        const auto preview_width = runtime["preview_tile_width"];
        const auto preview_height = runtime["preview_tile_height"];
        if (!preview_width.empty()) preview_width >> config.preview_tile_width;
        if (!preview_height.empty()) preview_height >> config.preview_tile_height;
        runtime["inference_interval"] >> config.inference_interval;
        std::string event_log;
        runtime["event_log"] >> event_log;
        if (!event_log.empty()) config.event_log = event_log;
    }
    config.preview_tile_width = std::max(1, config.preview_tile_width);
    config.preview_tile_height = std::max(1, config.preview_tile_height);
    config.inference_interval = std::max(1, config.inference_interval);
    return config;
}

}  // namespace seatvision::product
