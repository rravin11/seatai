#pragma once

#include <filesystem>
#include <fstream>

#include "seatvision/product/types.hpp"

namespace seatvision::product {

class EventSink {
public:
    explicit EventSink(const std::filesystem::path& path);
    void publish(const SeatEvent& event);

private:
    std::ofstream output_;
};

}  // namespace seatvision::product
