#include "jinja-lexer.h"
#include "jinja-vm.h"

#include <vector>
#include <string>
#include <map>
#include <regex>
#include <stdexcept>
#include <cctype>
#include <functional>
#include <string_view>

#define FILENAME "jinja-lexer"

namespace jinja {

// Trim template markers with '-' for whitespace control
// Example: [spaces]{%- ... -%} --> {% ... %}
#include <string>
#include <cctype>

static void trim_template_markers_inplace(std::string & s) {
    // i = head ; j = tail (i <= j)
    size_t j = 0; // Write pointer
    const size_t len = s.length();
    
    for (size_t i = 0; i < len; ) {
        bool handled = false;

        // We need at least 3 characters for any marker: {X- or -X}
        if (i + 2 < len) {
            const char c1 = s[i];
            const char c2 = s[i + 1];
            const char c3 = s[i + 2];

            // 1. Closing trim: -X} where X = %, }, #
            // Example: [content]-%} [spaces] -> [content]%}
            if (c1 == '-' && c3 == '}' && (c2 == '%' || c2 == '}' || c2 == '#')) {
                s[j++] = c2;
                s[j++] = '}';
                i += 3;
                // Strip leading whitespace AFTER the tag
                while (i < len && std::isspace(static_cast<unsigned char>(s[i]))) {
                    i++;
                }
                handled = true;
            }
            // 2. Opening trim: {X- where X = %, {, #
            // Example: [spaces]{%- [content] -> {% [content]
            else if (c1 == '{' && c3 == '-' && (c2 == '%' || c2 == '{' || c2 == '#')) {
                // Trim trailing whitespace BEFORE the tag by moving write pointer back
                while (j > 0 && std::isspace(static_cast<unsigned char>(s[j - 1]))) {
                    j--;
                }

                // Safety: Prevent merging '{' with tag start (avoid creating '{{%' or '{{{')
                // if the character immediately before our new tag is a literal '{'.
                if (j > 0 && s[j - 1] == '{') {
                    s[j++] = ' ';
                }

                s[j++] = '{';
                s[j++] = c2;
                i += 3;
                handled = true;
            }
        }

        if (!handled) {
            // Note: j is always <= i here, so this is safe.
            s[j++] = s[i++];
        }
    }

    s.resize(j);
}

std::string lexer::preprocess(const std::string & template_str, const preprocess_options & options) const {
    std::string result = template_str;
    // According to https://jinja.palletsprojects.com/en/3.0.x/templates/#whitespace-control

    // In the default configuration:
    //  - a single trailing newline is stripped if present
    //  - other whitespace (spaces, tabs, newlines etc.) is returned unchanged
    if (!result.empty() && result.back() == '\n') {
        result.pop_back();
    }

    if (options.lstrip_blocks) {
        // The lstrip_blocks option can also be set to strip tabs and spaces from the
        // beginning of a line to the start of a block. (Nothing will be stripped if
        // there are other characters before the start of the block.)
        // result = std::regex_replace(result, std::regex(R"((?m)^[ \t]*(\{[#%-]))"), "$1");
        throw std::runtime_error("lstrip_blocks option is not implemented yet");
    }

    if (options.trim_blocks) {
        // If an application configures Jinja to trim_blocks, the first newline after
        // a template tag is removed automatically (like in PHP).
        result = std::regex_replace(result, std::regex(R"(([#%-]\})\n)"), "$1");
    }

    // Handle whitespace control with - in tags
    trim_template_markers_inplace(result);

    // Handle custom transformers-specific `generation` tag
    // See https://github.com/huggingface/transformers/pull/30650 for more information.
    result = std::regex_replace(result, std::regex(R"(\{%\s*generation\s*%\})"), "");
    result = std::regex_replace(result, std::regex(R"(\{%\s*endgeneration\s*%\})"), "");

    return result;
}

lexer_result lexer::tokenize(const std::string & input, const preprocess_options & options) {
    std::vector<token> tokens;
    std::string src = preprocess(input, options);
    JJ_DEBUG("preprocessed input: '%s'", src.c_str());

    size_t pos = 0;
    size_t start_pos = 0;
    size_t curly_bracket_depth = 0;

    using pred = std::function<bool(char)>;
    auto consume_while = [&](pred predicate) -> std::string {
        std::string str;
        while (predicate(src[pos])) {
            // check for escape char
            if (src[pos] == '\\') {
                // consume backslash
                ++pos;
                // check for end of input
                if (pos >= src.size()) {
                    throw std::runtime_error("lexer: unexpected end of input after escape character");
                }
                // add escaped char
                char escaped_char = src[pos++];
                if (escape_chars.find(escaped_char) == escape_chars.end()) {
                    throw std::runtime_error(std::string("lexer: unknown escape character \\") + escaped_char);
                }
                char unescaped_char = escape_chars.at(escaped_char);
                str += unescaped_char;
                continue;
            }

            str += src[pos++];
            if (pos > src.size()) {
                throw std::runtime_error("lexer: unexpected end of input during consume_while");
            }
        }
        return str;
    };

    auto next_pos_is = [&](std::initializer_list<char> chars) -> bool {
        if (pos + 1 >= src.size()) return false;
        for (char c : chars) {
            if (src[pos + 1] == c) return true;
        }
        return false;
    };

    while (pos < src.size()) {
        start_pos = pos;
        JJ_DEBUG("lexer main loop at pos %zu: '%s...'", pos, src.substr(pos, 10).c_str());

        // First, consume all text that is outside of a Jinja statement or expression
        token::type last_token_type = tokens.empty()
                                            ? token::undefined
                                            : tokens.back().t;
        if (last_token_type == token::undefined ||
            last_token_type == token::close_statement ||
            last_token_type == token::close_expression ||
            last_token_type == token::comment) {
            std::string text;
            while (pos < src.size() &&
                    // Keep going until we hit the next Jinja statement or expression
                    !(
                        src[pos] == '{' &&
                        next_pos_is( {'%', '{', '#'} )
                    )) {
                text += src[pos++];
            }
            JJ_DEBUG("consumed text: '%s'", text.c_str());
            if (!text.empty()) {
                tokens.push_back({token::text, text, start_pos});
                continue;
            }
        }

        // Possibly consume a comment
        if (src[pos] == '{' && next_pos_is( {'#'} )) {
            start_pos = pos;
            pos += 2; // Skip the opening {#
            std::string comment;
            while (!(src[pos] == '#' && next_pos_is( {'}'} ))) {
                if (pos + 2 >= src.size()) {
                    throw std::runtime_error("lexer: missing end of comment tag");
                }
                comment += src[pos++];
            }
            JJ_DEBUG("consumed comment: '%s'", comment.c_str());
            tokens.push_back({token::comment, comment, start_pos});
            pos += 2; // Skip the closing #}
            continue;
        }

        // Consume (and ignore) all whitespace inside Jinja statements or expressions
        consume_while([](char c) { return std::isspace(static_cast<unsigned char>(c)); });

        if (pos >= src.size()) break;

        char ch = src[pos];

        // Check for unary operators
        if (ch == '-' || ch == '+') {
            start_pos = pos;
            token::type last_token_type = tokens.empty() ? token::undefined : tokens.back().t;
            if (last_token_type == token::text || last_token_type == token::undefined) {
                throw std::runtime_error(std::string("lexer: unexpected character: ") + ch);
            }
            switch (last_token_type) {
                case token::identifier:
                case token::numeric_literal:
                case token::string_literal:
                case token::close_paren:
                case token::close_square_bracket:
                    // Part of a binary operator
                    // a - 1, 1 - 1, true - 1, "apple" - 1, (1) - 1, a[1] - 1
                    // Continue parsing normally
                    break;
                default: {
                    // Is part of a unary operator
                    // (-1), [-1], (1 + -1), not -1, -apple
                    ++pos; // Consume the operator

                    // Check for numbers following the unary operator
                    std::string num = consume_while(is_integer);
                    std::string value = std::string(1, ch) + num;
                    token::type t = num.empty() ? token::unary_operator : token::numeric_literal;
                    JJ_DEBUG("consumed unary operator or numeric literal: '%s'", value.c_str());
                    tokens.push_back({t, value, start_pos});
                    continue;
                }
            }
        }

        // Try to match one of the tokens in the mapping table
        bool matched = false;
        for (const auto & [seq, typ] : ordered_mapping_table) {
            start_pos = pos;
            // Inside an object literal, don't treat "}}" as expression-end
            if (seq == "}}" && curly_bracket_depth > 0) {
                continue;
            }
            if (pos + seq.size() <= src.size() && src.substr(pos, seq.size()) == seq) {
                tokens.push_back({typ, seq, start_pos});
                if (typ == token::open_expression) {
                    curly_bracket_depth = 0;
                } else if (typ == token::open_curly_bracket) {
                    ++curly_bracket_depth;
                } else if (typ == token::close_curly_bracket) {
                    --curly_bracket_depth;
                }
                pos += seq.size();
                matched = true;
                break; // continue main loop
            }
        }
        if (matched) continue; // continue main loop

        // Strings
        if (ch == '\'' || ch == '"') {
            start_pos = pos;
            ++pos; // Skip opening quote
            std::string str = consume_while([ch](char c) { return c != ch; });
            tokens.push_back({token::string_literal, str, start_pos});
            ++pos; // Skip closing quote
            continue;
        }

        // Numbers
        if (is_integer(ch)) {
            start_pos = pos;
            std::string num = consume_while(is_integer);
            if (pos < src.size() && src[pos] == '.' && pos + 1 < src.size() && is_integer(src[pos + 1])) {
                ++pos; // Consume '.'
                std::string frac = consume_while(is_integer);
                num += "." + frac;
            }
            tokens.push_back({token::numeric_literal, num, start_pos});
            continue;
        }

        // Identifiers
        if (is_word(ch)) {
            start_pos = pos;
            std::string word = consume_while(is_word);
            tokens.push_back({token::identifier, word, start_pos});
            continue;
        }

        throw std::runtime_error(std::string("lexer: unexpected character: ") + ch);
    }

    return {std::move(tokens), std::move(src)};
}

} // namespace jinja
