#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>

#include "seatvision/argus_camera.hpp"
#include "seatvision/product/config.hpp"
#include "seatvision/product/types.hpp"

namespace seatvision::product {

class CameraStream {
public:
    explicit CameraStream(CameraConfig config);
    ~CameraStream();
    CameraStream(const CameraStream&) = delete;
    CameraStream& operator=(const CameraStream&) = delete;

    // Argus camera discovery is serialized: prepare every sensor before any
    // capture worker begins requesting frames. This avoids an Argus startup race.
    bool prepare();
    bool start();
    void stop();
    std::optional<Frame> latest_after(std::uint64_t sequence) const;
    [[nodiscard]] int camera_id() const { return config_.sensor_id; }

private:
    void run();
    CameraConfig config_;
    ArgusCamera camera_;
    mutable std::mutex mutex_;
    std::optional<Frame> latest_;
    bool prepared_{false};
    std::atomic<bool> running_{false};
    std::thread worker_;
};

}  // namespace seatvision::product
