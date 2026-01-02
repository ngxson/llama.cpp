#pragma once

#include <vector>

#include "jinja-value.h"
#include "jinja-vm.h"

namespace jinja {

struct caps {
    bool supports_tools = true;
    bool supports_tool_calls = true;
    bool supports_system_role = true;
    bool supports_parallel_tool_calls = true;
    bool requires_typed_content = false; // default: use string content
};

caps caps_get(jinja::program & prog);
void debug_print_caps(const caps & c);

} // namespace jinja
