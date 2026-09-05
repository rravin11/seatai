#include "seatvision/product/event_sink.hpp"

#include <chrono>
#include <iomanip>
#include <stdexcept>

namespace seatvision::product {
namespace {
const char* name_for(Occupancy state) {
    switch (state) {
        case Occupancy::Available: return "available";
        case Occupancy::Occupied: return "occupied";
        case Occupancy::Claimed: return "claimed";
        case Occupancy::Occluded: return "occluded";
        default: return "unknown";
    }
}
}  // namespace

EventSink::EventSink(const std::filesystem::path& path) {
    std::filesystem::create_directories(path.parent_path());
    output_.open(path, std::ios::app);
    if (!output_) throw std::runtime_error("Unable to open event log: " + path.string());
}

void EventSink::publish(const SeatEvent& event) {
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(event.timestamp.time_since_epoch()).count();
    output_ << "{\"type\":\"" << event.type << "\",\"seat_id\":\"" << event.seat_id
            << "\",\"camera_id\":" << event.camera_id << ",\"from\":\"" << name_for(event.from)
            << "\",\"to\":\"" << name_for(event.to) << "\",\"confidence\":" << std::fixed << std::setprecision(3)
            << event.confidence << ",\"steady_time_ms\":" << milliseconds << "}\n";
    output_.flush();
}

}  // namespace seatvision::product
