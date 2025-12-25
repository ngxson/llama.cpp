#include <vector>
#include <string>
#include <map>
#include <regex>
#include <stdexcept>
#include <cctype>
#include <functional>

// #define JJ_DEBUG(msg, ...)  printf("jinja-lexer: " msg "\n", __VA_ARGS__)
#define JJ_DEBUG(msg, ...)  // no-op

namespace jinja {

struct preprocess_options {
    bool trim_blocks = false;
    bool lstrip_blocks = false;
};

struct token {
    enum type {
        undefined,
        text, // The text between Jinja statements or expressions

        numeric_literal, // e.g., 123, 1.0
        string_literal, // 'string'
        identifier, // Variables, functions, statements, booleans, etc.
        equals, // =
        open_paren, // (
        close_paren, // )
        open_statement, // {%
        close_statement, // %}
        open_expression, // {{
        close_expression, // }}
        open_square_bracket, // [
        close_square_bracket, // ]
        open_curly_bracket, // {
        close_curly_bracket, // }
        comma, // ,
        dot, // .
        colon, // :
        pipe, // |

        call_operator, // ()
        additive_binary_operator, // + - ~
        multiplicative_binary_operator, // * / %
        comparison_binary_operator, // < > <= >= == !=
        unary_operator, // ! - +
        comment, // {# ... #}
    };
    type t;
    std::string value;
};

struct lexer {
    const std::map<char, char> escape_chars = {
        {'n', '\n'},
        {'t', '\t'},
        {'r', '\r'},
        {'b', '\b'},
        {'f', '\f'},
        {'v', '\v'},
        {'\\', '\\'},
        {'\'', '\''},
        {'\"', '\"'},
    };

    static bool is_word(char c) {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
    }

    static bool is_integer(char c) {
        return std::isdigit(static_cast<unsigned char>(c));
    }

    const std::vector<std::pair<std::string, token::type>> ordered_mapping_table = {
        // Control sequences
        {"{%", token::open_statement},
        {"%}", token::close_statement},
        {"{{", token::open_expression},
        {"}}", token::close_expression},
        // Single character tokens
        {"(", token::open_paren},
        {")", token::close_paren},
        {"{", token::open_curly_bracket},
        {"}", token::close_curly_bracket},
        {"[", token::open_square_bracket},
        {"]", token::close_square_bracket},
        {",", token::comma},
        {".", token::dot},
        {":", token::colon},
        {"|", token::pipe},
        // Comparison operators
        {"<=", token::comparison_binary_operator},
        {">=", token::comparison_binary_operator},
        {"==", token::comparison_binary_operator},
        {"!=", token::comparison_binary_operator},
        {"<", token::comparison_binary_operator},
        {">", token::comparison_binary_operator},
        // Arithmetic operators
        {"+", token::additive_binary_operator},
        {"-", token::additive_binary_operator},
        {"~", token::additive_binary_operator},
        {"*", token::multiplicative_binary_operator},
        {"/", token::multiplicative_binary_operator},
        {"%", token::multiplicative_binary_operator},
        // Assignment operator
        {"=", token::equals},
    };

    std::string preprocess(const std::string& template_str, const preprocess_options& options) const {
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
        result = std::regex_replace(result, std::regex(R"(-%\}\s*)"), "%}");
        result = std::regex_replace(result, std::regex(R"(\s*\{%-)"), "{%");
        result = std::regex_replace(result, std::regex(R"(-\}\}\s*)"), "}}");
        result = std::regex_replace(result, std::regex(R"(\s*\{\{-)"), "{{");
        result = std::regex_replace(result, std::regex(R"(-#\}\s*)"), "#}");
        result = std::regex_replace(result, std::regex(R"(\s*\{\#-)"), "{#");

        // Handle custom transformers-specific `generation` tag
        // See https://github.com/huggingface/transformers/pull/30650 for more information.
        // result = std::regex_replace(result, std::regex(R"((?s)\{%\s*generation\s*%\}.+?\{%\s*endgeneration\s*%\})"), "");

        return result;
    }

    std::vector<token> tokenize(const std::string & input, const preprocess_options & options = {}) {
        std::vector<token> tokens;
        std::string src = preprocess(input, options);
        JJ_DEBUG("preprocessed input: '%s'", src.c_str());

        size_t pos = 0;
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
                    tokens.push_back({token::text, text});
                    continue;
                }
            }

            // Possibly consume a comment
            if (src[pos] == '{' && next_pos_is( {'#'} )) {
                pos += 2; // Skip the opening {#
                std::string comment;
                while (!(src[pos] == '#' && next_pos_is( {'}'} ))) {
                    if (pos + 2 >= src.size()) {
                        throw std::runtime_error("lexer: missing end of comment tag");
                    }
                    comment += src[pos++];
                }
                JJ_DEBUG("consumed comment: '%s'", comment.c_str());
                tokens.push_back({token::comment, comment});
                pos += 2; // Skip the closing #}
                continue;
            }

            // Consume (and ignore) all whitespace inside Jinja statements or expressions
            consume_while([](char c) { return std::isspace(static_cast<unsigned char>(c)); });

            if (pos >= src.size()) break;

            char ch = src[pos];

            // Check for unary operators
            if (ch == '-' || ch == '+') {
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
                        tokens.push_back({t, value});
                        continue;
                    }
                }
            }

            // Try to match one of the tokens in the mapping table
            bool matched = false;
            for (const auto & [seq, typ] : ordered_mapping_table) {
                // Inside an object literal, don't treat "}}" as expression-end
                if (seq == "}}" && curly_bracket_depth > 0) {
                    continue;
                }
                if (pos + seq.size() <= src.size() && src.substr(pos, seq.size()) == seq) {
                    tokens.push_back({typ, seq});
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
                ++pos; // Skip opening quote
                std::string str = consume_while([ch](char c) { return c != ch; });
                tokens.push_back({token::string_literal, str});
                ++pos; // Skip closing quote
                continue;
            }

            // Numbers
            if (is_integer(ch)) {
                std::string num = consume_while(is_integer);
                if (pos < src.size() && src[pos] == '.' && pos + 1 < src.size() && is_integer(src[pos + 1])) {
                    ++pos; // Consume '.'
                    std::string frac = consume_while(is_integer);
                    num += "." + frac;
                }
                tokens.push_back({token::numeric_literal, num});
                continue;
            }

            // Identifiers
            if (is_word(ch)) {
                std::string word = consume_while(is_word);
                tokens.push_back({token::identifier, word});
                continue;
            }

            throw std::runtime_error(std::string("lexer: unexpected character: ") + ch);
        }

        return tokens;
    }
};

} // namespace jinja
