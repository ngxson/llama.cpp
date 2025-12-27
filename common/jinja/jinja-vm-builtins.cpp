#include "jinja-lexer.h"
#include "jinja-vm.h"
#include "jinja-parser.h"
#include "jinja-value.h"

#include <string>
#include <cctype>

namespace jinja {

static std::string string_strip(const std::string & str, bool left, bool right) {
    size_t start = 0;
    size_t end = str.length();
    if (left) {
        while (start < end && isspace(static_cast<unsigned char>(str[start]))) {
            ++start;
        }
    }
    if (right) {
        while (end > start && isspace(static_cast<unsigned char>(str[end - 1]))) {
            --end;
        }
    }
    return str.substr(start, end - start);
}

static bool string_startswith(const std::string & str, const std::string & prefix) {
    if (str.length() < prefix.length()) return false;
    return str.compare(0, prefix.length(), prefix) == 0;
}

static bool string_endswith(const std::string & str, const std::string & suffix) {
    if (str.length() < suffix.length()) return false;
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

const func_builtins & value_string_t::get_builtins() const {
    static const func_builtins builtins = {
        {"upper", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            std::string str = args.args[0]->as_string();
            std::transform(str.begin(), str.end(), str.begin(), ::toupper);
            return std::make_unique<value_string_t>(str);
        }},
        {"lower", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            std::string str = args.args[0]->as_string();
            std::transform(str.begin(), str.end(), str.begin(), ::tolower);
            return std::make_unique<value_string_t>(str);
        }},
        {"strip", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            std::string str = args.args[0]->as_string();
            return std::make_unique<value_string_t>(string_strip(str, true, true));
        }},
        {"rstrip", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            std::string str = args.args[0]->as_string();
            return std::make_unique<value_string_t>(string_strip(str, false, true));
        }},
        {"lstrip", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            std::string str = args.args[0]->as_string();
            return std::make_unique<value_string_t>(string_strip(str, true, false));
        }},
        {"title", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            std::string str = args.args[0]->as_string();
            bool capitalize_next = true;
            for (char &c : str) {
                if (isspace(static_cast<unsigned char>(c))) {
                    capitalize_next = true;
                } else if (capitalize_next) {
                    c = ::toupper(static_cast<unsigned char>(c));
                    capitalize_next = false;
                } else {
                    c = ::tolower(static_cast<unsigned char>(c));
                }
            }
            return std::make_unique<value_string_t>(str);
        }},
        {"capitalize", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            std::string str = args.args[0]->as_string();
            if (!str.empty()) {
                str[0] = ::toupper(static_cast<unsigned char>(str[0]));
                std::transform(str.begin() + 1, str.end(), str.begin() + 1, ::tolower);
            }
            return std::make_unique<value_string_t>(str);
        }},
        {"length", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            std::string str = args.args[0]->as_string();
            return std::make_unique<value_int_t>(str.length());
        }},
        {"startswith", [](const func_args & args) -> value {
            args.ensure_vals<value_string, value_string>();
            std::string str = args.args[0]->as_string();
            std::string prefix = args.args[1]->as_string();
            return std::make_unique<value_bool_t>(string_startswith(str, prefix));
        }},
        {"endswith", [](const func_args & args) -> value {
            args.ensure_vals<value_string, value_string>();
            std::string str = args.args[0]->as_string();
            std::string suffix = args.args[1]->as_string();
            return std::make_unique<value_bool_t>(string_endswith(str, suffix));
        }},
        {"split", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            std::string str = args.args[0]->as_string();
            std::string delim = (args.args.size() > 1) ? args.args[1]->as_string() : " ";
            auto result = std::make_unique<value_array_t>();
            size_t pos = 0;
            std::string token;
            while ((pos = str.find(delim)) != std::string::npos) {
                token = str.substr(0, pos);
                result->val_arr->push_back(std::make_unique<value_string_t>(token));
                str.erase(0, pos + delim.length());
            }
            result->val_arr->push_back(std::make_unique<value_string_t>(str));
            return std::move(result);
        }},
        {"replace", [](const func_args & args) -> value {
            args.ensure_vals<value_string, value_string, value_string>();
            std::string str = args.args[0]->as_string();
            std::string old_str = args.args[1]->as_string();
            std::string new_str = args.args[2]->as_string();
            size_t pos = 0;
            while ((pos = str.find(old_str, pos)) != std::string::npos) {
                str.replace(pos, old_str.length(), new_str);
                pos += new_str.length();
            }
            return std::make_unique<value_string_t>(str);
        }},
    };
    return builtins;
};

} // namespace jinja
