#include "seatvision/product/semantics.hpp"

namespace seatvision::product {

EntityRole classify_label(std::string_view label, const SemanticPolicy& semantics) {
    const std::string owned_label{label};
    if (semantics.ignored_labels.contains(owned_label)) return EntityRole::Ignored;
    if (semantics.people_labels.contains(owned_label)) return EntityRole::Person;
    if (semantics.seat_labels.contains(owned_label)) return EntityRole::Seat;
    return EntityRole::Object;
}

const char* name_for(EntityRole role) {
    switch (role) {
        case EntityRole::Person: return "person";
        case EntityRole::Seat: return "seat";
        case EntityRole::Object: return "object";
        case EntityRole::Ignored: return "ignored";
        default: return "unknown";
    }
}

}  // namespace seatvision::product
