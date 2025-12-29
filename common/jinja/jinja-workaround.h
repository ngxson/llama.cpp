#pragma once

#include "jinja-value.h"

#include <string>
#include <vector>

namespace jinja {

// containing workarounds for Jinja templates that rely on non-standard behavior
// NOTE: this is kept as a dedicated file for better documentation

struct workarounds {
    // meetkai-functionary-medium-v3.1.jinja call filter on None type
    bool none_has_builtins = true;

    // Olmo calls operation + between string and undefined
    bool string_plus_undefined_is_string = true;

    // sheldonrobinson-Llama-Guard call selectattr on string
    bool string_has_selectattr = true;
};

} // namespace jinja
