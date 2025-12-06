// Console functions

#pragma once

#include <string>

namespace console {
    enum display_t {
        reset = 0,
        info,
        prompt,
        user_input,
        error
    };

    void init(bool use_simple_io, bool use_advanced_display);
    void cleanup();
    void set_display(display_t display);
    void set_loading(bool enabled); // update frame each time it's called
    bool readline(std::string & line, bool multiline_input);
}
