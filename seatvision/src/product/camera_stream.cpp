#include "seatvision/product/camera_stream.hpp"

#include <chrono>
#include <iostream>

namespace seatvision::product {

CameraStream::CameraStream(CameraConfig config)
    : config_(config), camera_(config.sensor_id, config.width, config.height, config.fps, config.rotate_180) {}

CameraStream::~CameraStream() { stop(); }

bool CameraStream::prepare() {
    if (prepared_) return true;
    prepared_ = camera_.open();
    return prepared_;
}

bool CameraStream::start() {
    if (running_) return true;
    if (!prepare()) return false;
    running_ = true;
    worker_ = std::thread(&CameraStream::run, this);
    return true;
}

void CameraStream::stop() {
    running_ = false;
    if (worker_.joinable()) worker_.join();
}

void CameraStream::run() {
    std::uint64_t sequence{};
    const auto started_at = Clock::now();
    auto last_error_log = started_at;
    std::uint32_t consecutive_read_failures{};
    bool received_first_frame{};
    while (running_) {
        cv::Mat image;
        if (!camera_.read(image)) {
            ++consecutive_read_failures;
            const auto now = Clock::now();
            const bool startup_timeout = !received_first_frame && now - started_at >= std::chrono::seconds(3);
            const bool sustained_failure = received_first_frame && consecutive_read_failures >= 5;
            if ((startup_timeout || sustained_failure) && now - last_error_log >= std::chrono::seconds(1)) {
                std::cerr << "Camera " << config_.sensor_id << " lost frames (capture pipeline unhealthy).\n";
                last_error_log = now;
            }
            // Avoid a tight spin if Argus temporarily has no buffer. This also
            // leaves CPU/GPU time for the other sensor and inference workers.
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        consecutive_read_failures = 0;
        received_first_frame = true;
        Frame frame{config_.sensor_id, ++sequence, Clock::now(), std::move(image)};
        std::lock_guard lock(mutex_);
        latest_ = std::move(frame);
    }
}

std::optional<Frame> CameraStream::latest_after(std::uint64_t sequence) const {
    std::lock_guard lock(mutex_);
    if (!latest_.has_value() || latest_->sequence <= sequence) return std::nullopt;
    Frame copy = *latest_;
    copy.image = latest_->image.clone();
    return copy;
}

}  // namespace seatvision::product
