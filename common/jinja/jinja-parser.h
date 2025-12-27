#pragma once

#include "jinja-lexer.h"
#include "jinja-vm.h"

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <algorithm>

namespace jinja {

program parse_from_tokens(const std::vector<token> & tokens);

} // namespace jinja
