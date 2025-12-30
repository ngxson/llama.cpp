#pragma once

#include <memory>
#include <string>

#include "jinja-value.h"

namespace jinja {

struct value_t;
using value = std::shared_ptr<value_t>;

// this is used as a hint for chat parsing
// it is not a 1-to-1 mapping to value_t derived types
enum class inferred_type {
    numeric, // int, float
    string,
    boolean,
    array,
    object,
    optional, // null, undefined
    unknown,
};

static std::string inferred_type_to_string(inferred_type type) {
    switch (type) {
        case inferred_type::numeric: return "numeric";
        case inferred_type::string: return "string";
        case inferred_type::boolean: return "boolean";
        case inferred_type::array: return "array";
        case inferred_type::object: return "object";
        case inferred_type::optional: return "optional";
        case inferred_type::unknown: return "unknown";
        default: return "invalid";
    }
}

} // namespace jinja
