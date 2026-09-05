#pragma once

#include <string_view>

#include "seatvision/product/config.hpp"
#include "seatvision/product/types.hpp"

namespace seatvision::product {

// Maps deployment taxonomy data to a stable product role. Keeping this in one
// place lets live and offline tools report the same semantics without embedding
// model labels in either executable.
EntityRole classify_label(std::string_view label, const SemanticPolicy& semantics);
const char* name_for(EntityRole role);

}  // namespace seatvision::product
