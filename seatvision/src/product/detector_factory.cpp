#include "seatvision/product/detector.hpp"

#include <stdexcept>

namespace seatvision::product {
std::unique_ptr<Detector> create_opencv_yolo_detector(const DetectorConfig& config);
std::unique_ptr<Detector> create_tensorrt_yolo_detector(const DetectorConfig& config);

std::unique_ptr<Detector> create_detector(const DetectorConfig& config) {
    if (config.labels.empty()) throw std::runtime_error("Detector labels must be supplied by the model manifest/configuration.");
    if (config.backend == "tensorrt") return create_tensorrt_yolo_detector(config);
    if (config.backend == "opencv") return create_opencv_yolo_detector(config);
    throw std::runtime_error("Unsupported detector backend: " + config.backend);
}
}  // namespace seatvision::product
