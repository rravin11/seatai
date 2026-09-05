#include <exception>
#include <iostream>
#include <string>

#include "seatvision/product/config.hpp"
#include "seatvision/product/runtime.hpp"

int main(int argc, char** argv) {
    std::string config_path{"config/jetson.yaml"};
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--config" && i + 1 < argc) config_path = argv[++i];
        else if (argument == "--headless") { /* applied after loading config */ }
        else if (argument == "--help") { std::cout << "seatvisiond [--config config/jetson.yaml] [--headless]\n"; return 0; }
        else { std::cerr << "Unknown or incomplete argument: " << argument << '\n'; return 2; }
    }
    try {
        auto config = seatvision::product::load_config(config_path);
        for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--headless") config.show_window = false;
        return seatvision::product::Runtime(std::move(config)).run();
    } catch (const std::exception& error) {
        std::cerr << "SeatVision product startup failed: " << error.what() << '\n';
        return 1;
    }
}
