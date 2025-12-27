#include "jinja-lexer.h"
#include "jinja-vm.h"
#include "jinja-parser.h"
#include "jinja-value.h"

#include <string>
#include <cctype>

namespace jinja {

const func_builtins & value_int_t::get_builtins() const {
    static const func_builtins builtins = {
        {"abs", [](const func_args & args) -> value {
            args.ensure_vals<value_int>();
            int64_t val = args.args[0]->as_int();
            return std::make_unique<value_int_t>(val < 0 ? -val : val);
        }},
        {"float", [](const func_args & args) -> value {
            args.ensure_vals<value_int>();
            double val = static_cast<double>(args.args[0]->as_int());
            return std::make_unique<value_float_t>(val);
        }},
    };
    return builtins;
}


const func_builtins & value_float_t::get_builtins() const {
    static const func_builtins builtins = {
        {"abs", [](const func_args & args) -> value {
            args.ensure_vals<value_float>();
            double val = args.args[0]->as_float();
            return std::make_unique<value_float_t>(val < 0.0 ? -val : val);
        }},
        {"int", [](const func_args & args) -> value {
            args.ensure_vals<value_float>();
            int64_t val = static_cast<int64_t>(args.args[0]->as_float());
            return std::make_unique<value_int_t>(val);
        }},
    };
    return builtins;
}


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
        {"int", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            std::string str = args.args[0]->as_string();
            try {
                return std::make_unique<value_int_t>(std::stoi(str));
            } catch (...) {
                throw std::runtime_error("Cannot convert string '" + str + "' to int");
            }
        }},
        {"float", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            std::string str = args.args[0]->as_string();
            try {
                return std::make_unique<value_float_t>(std::stod(str));
            } catch (...) {
                throw std::runtime_error("Cannot convert string '" + str + "' to float");
            }
        }},
        {"string", [](const func_args & args) -> value {
            // no-op
            args.ensure_vals<value_string>();
            return std::make_unique<value_string_t>(args.args[0]->as_string());
        }},
        {"indent", [](const func_args & args) -> value {
            throw std::runtime_error("indent builtin not implemented");
        }},
        {"join", [](const func_args & args) -> value {
            throw std::runtime_error("join builtin not implemented");
        }},
    };
    return builtins;
};


const func_builtins & value_bool_t::get_builtins() const {
    static const func_builtins builtins = {
        {"int", [](const func_args & args) -> value {
            args.ensure_vals<value_bool>();
            bool val = args.args[0]->as_bool();
            return std::make_unique<value_int_t>(val ? 1 : 0);
        }},
        {"float", [](const func_args & args) -> value {
            args.ensure_vals<value_bool>();
            bool val = args.args[0]->as_bool();
            return std::make_unique<value_float_t>(val ? 1.0 : 0.0);
        }},
        {"string", [](const func_args & args) -> value {
            args.ensure_vals<value_bool>();
            bool val = args.args[0]->as_bool();
            return std::make_unique<value_string_t>(val ? "True" : "False");
        }},
    };
    return builtins;
}


const func_builtins & value_array_t::get_builtins() const {
    static const func_builtins builtins = {
        {"list", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            const auto & arr = args.args[0]->as_array();
            auto result = std::make_unique<value_array_t>();
            for (const auto& v : arr) {
                result->val_arr->push_back(v->clone());
            }
            return result;
        }},
        {"first", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            const auto & arr = args.args[0]->as_array();
            if (arr.empty()) {
                return std::make_unique<value_undefined_t>();
            }
            return arr[0]->clone();
        }},
        {"last", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            const auto & arr = args.args[0]->as_array();
            if (arr.empty()) {
                return std::make_unique<value_undefined_t>();
            }
            return arr[arr.size() - 1]->clone();
        }},
        {"length", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            const auto & arr = args.args[0]->as_array();
            return std::make_unique<value_int_t>(static_cast<int64_t>(arr.size()));
        }},
        // TODO: reverse, sort, join, string, unique
    };
    return builtins;
}


const func_builtins & value_object_t::get_builtins() const {
    static const func_builtins builtins = {
        {"get", [](const func_args & args) -> value {
            args.ensure_vals<value_object, value_string>(); // TODO: add default value
            const auto & obj = args.args[0]->as_object();
            std::string key = args.args[1]->as_string();
            auto it = obj.find(key);
            if (it != obj.end()) {
                return it->second->clone();
            } else {
                return std::make_unique<value_undefined_t>();
            }
        }},
        {"keys", [](const func_args & args) -> value {
            args.ensure_vals<value_object>();
            const auto & obj = args.args[0]->as_object();
            auto result = std::make_unique<value_array_t>();
            for (const auto & pair : obj) {
                result->val_arr->push_back(std::make_unique<value_string_t>(pair.first));
            }
            return result;
        }},
        {"values", [](const func_args & args) -> value {
            args.ensure_vals<value_object>();
            const auto & obj = args.args[0]->as_object();
            auto result = std::make_unique<value_array_t>();
            for (const auto & pair : obj) {
                result->val_arr->push_back(pair.second->clone());
            }
            return result;
        }},
        {"items", [](const func_args & args) -> value {
            args.ensure_vals<value_object>();
            const auto & obj = args.args[0]->as_object();
            auto result = std::make_unique<value_array_t>();
            for (const auto & pair : obj) {
                auto item = std::make_unique<value_array_t>();
                item->val_arr->push_back(std::make_unique<value_string_t>(pair.first));
                item->val_arr->push_back(pair.second->clone());
                result->val_arr->push_back(std::move(item));
            }
            return result;
        }},
    };
    return builtins;
}


} // namespace jinja
