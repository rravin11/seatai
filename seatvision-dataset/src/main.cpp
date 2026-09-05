#include <charconv>
#include <exception>
#include <iostream>
#include <string>
#include <string_view>

#include "seatvision_dataset/batch.hpp"
#include "seatvision_dataset/config.hpp"

namespace {

void print_help() {
    std::cout
        << "seatvision-dataset --input IMAGE_OR_DIRECTORY --output RUN_DIRECTORY [options]\n\n"
        << "Options:\n"
        << "  --config PATH              Dataset config (default: config/coco_yolov8n.yaml)\n"
        << "  --ground-truth PATH        COCO detection annotations; enables metrics.json\n"
        << "  --label-alias FROM=TO      Explicitly map a COCO category to a model label (repeatable)\n"
        << "  --limit N                  Process only the first N sorted images\n"
        << "  --non-recursive            Do not scan subdirectories\n"
        << "  --no-annotated             Skip annotated image writes\n"
        << "  --gallery-limit N          Maximum images in report.html (default: 120)\n"
        << "  --help                     Show this help\n\n"
        << "This tool validates static detections only; it never reports a seat occupancy state.\n";
}

bool parse_size(const std::string_view text, std::size_t& value) {
    unsigned long long parsed{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (error != std::errc{} || end != text.data() + text.size()) return false;
    value = static_cast<std::size_t>(parsed);
    return true;
}

bool parse_alias(const std::string& text, std::string& source, std::string& target) {
    const auto separator = text.find('=');
    if (separator == std::string::npos || separator == 0 || separator + 1 == text.size()) return false;
    source = text.substr(0, separator);
    target = text.substr(separator + 1);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string config_path{"config/coco_yolov8n.yaml"};
    seatvision::dataset::BatchOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            print_help();
            return 0;
        }
        if (argument == "--non-recursive") {
            options.recursive = false;
            continue;
        }
        if (argument == "--no-annotated") {
            options.save_annotated = false;
            continue;
        }
        if (argument == "--config" && index + 1 < argc) {
            config_path = argv[++index];
            continue;
        }
        if (argument == "--input" && index + 1 < argc) {
            options.input = argv[++index];
            continue;
        }
        if (argument == "--output" && index + 1 < argc) {
            options.output = argv[++index];
            continue;
        }
        if (argument == "--ground-truth" && index + 1 < argc) {
            options.coco_annotations = argv[++index];
            continue;
        }
        if (argument == "--label-alias" && index + 1 < argc) {
            std::string source;
            std::string target;
            if (!parse_alias(argv[++index], source, target)) {
                std::cerr << "--label-alias expects FROM=TO\n";
                return 2;
            }
            options.label_aliases.emplace(std::move(source), std::move(target));
            continue;
        }
        if ((argument == "--limit" || argument == "--gallery-limit") && index + 1 < argc) {
            std::size_t value{};
            if (!parse_size(argv[++index], value)) {
                std::cerr << argument << " expects a non-negative integer\n";
                return 2;
            }
            if (argument == "--limit") options.limit = value;
            else options.gallery_limit = value;
            continue;
        }
        std::cerr << "Unknown or incomplete argument: " << argument << '\n';
        print_help();
        return 2;
    }
    try {
        const auto config = seatvision::dataset::load_config(config_path);
        return seatvision::dataset::run_batch(config, options);
    } catch (const std::exception& error) {
        std::cerr << "SeatVision dataset validation failed: " << error.what() << '\n';
        return 1;
    }
}
