#pragma once

#include "jinja-value.h"

#include <string>
#include <vector>

namespace jinja {

// containing workarounds for Jinja templates that rely on non-standard behavior

struct workarounds {
    // meetkai-functionary-medium-v3.1.jinja call filter on None type
    bool none_has_builtins = true;

    // Olmo calls operation + between string and undefined
    bool string_plus_undefined_is_string = true;
};

} // namespace jinja
