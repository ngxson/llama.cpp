#include "jinja-lexer.h"
#include "jinja-vm.h"

namespace jinja {

void parse(const std::vector<token> & tokens) {
    auto program = std::make_unique<jinja::program>();
    size_t current = 0;

    /**
     * Consume the next token if it matches the expected type, otherwise throw an error.
     * @param type The expected token type
     * @param error The error message to throw if the token does not match the expected type
     * @returns The consumed token
     */
    auto expect = [&](const token::type & type, const std::string & error) -> token {
        const auto & prev = tokens[current++];
        if (prev.t != type) {
            throw std::runtime_error("Parser Error: " + error + " (" + type_to_string(prev.t) + " != " + type_to_string(type) + ")");
        }
        return prev;
    };

    auto next_token = [&]() -> const token & {
        if (current >= tokens.size()) {
            return token{token::undefined, ""};
        }
        return tokens[current++];
    };

    auto expect_identifier = [&](const std::string & name) -> void {
        if (!is_identifier(name)) {
            throw std::runtime_error("Expected " + name);
        }
        ++current;
    };
}

}; // namespace jinja
