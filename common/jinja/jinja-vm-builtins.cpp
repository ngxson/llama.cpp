#include "jinja-lexer.h"
#include "jinja-vm.h"
#include "jinja-parser.h"
#include "jinja-value.h"

#include <string>
#include <cctype>

namespace jinja {

const func_builtins & global_builtins() {
    static const func_builtins builtins = {
        {"raise_exception", [](const func_args & args) -> value {
            args.ensure_count(1);
            std::string msg = args.args[0]->as_string().str();
            throw raised_exception("Jinja Exception: " + msg);
        }},
    };
    return builtins;
}


const func_builtins & value_int_t::get_builtins() const {
    static const func_builtins builtins = {
        {"abs", [](const func_args & args) -> value {
            args.ensure_vals<value_int>();
            int64_t val = args.args[0]->as_int();
            return mk_val<value_int>(val < 0 ? -val : val);
        }},
        {"float", [](const func_args & args) -> value {
            args.ensure_vals<value_int>();
            double val = static_cast<double>(args.args[0]->as_int());
            return mk_val<value_float>(val);
        }},
    };
    return builtins;
}


const func_builtins & value_float_t::get_builtins() const {
    static const func_builtins builtins = {
        {"abs", [](const func_args & args) -> value {
            args.ensure_vals<value_float>();
            double val = args.args[0]->as_float();
            return mk_val<value_float>(val < 0.0 ? -val : val);
        }},
        {"int", [](const func_args & args) -> value {
            args.ensure_vals<value_float>();
            int64_t val = static_cast<int64_t>(args.args[0]->as_float());
            return mk_val<value_int>(val);
        }},
    };
    return builtins;
}


// static std::string string_strip(const std::string & str, bool left, bool right) {
//     size_t start = 0;
//     size_t end = str.length();
//     if (left) {
//         while (start < end && isspace(static_cast<unsigned char>(str[start]))) {
//             ++start;
//         }
//     }
//     if (right) {
//         while (end > start && isspace(static_cast<unsigned char>(str[end - 1]))) {
//             --end;
//         }
//     }
//     return str.substr(start, end - start);
// }

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
            jinja::string str = args.args[0]->as_string().uppercase();
            return mk_val<value_string>(str);
        }},
        {"lower", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.args[0]->as_string().lowercase();
            return mk_val<value_string>(str);
        }},
        {"strip", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.args[0]->as_string().strip(true, true);
            return mk_val<value_string>(str);
        }},
        {"rstrip", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.args[0]->as_string().strip(false, true);
            return mk_val<value_string>(str);
        }},
        {"lstrip", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.args[0]->as_string().strip(true, false);
            return mk_val<value_string>(str);
        }},
        {"title", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.args[0]->as_string().titlecase();
            return mk_val<value_string>(str);
        }},
        {"capitalize", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.args[0]->as_string().capitalize();
            return mk_val<value_string>(str);
        }},
        {"length", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.args[0]->as_string();
            return mk_val<value_int>(str.length());
        }},
        {"startswith", [](const func_args & args) -> value {
            args.ensure_vals<value_string, value_string>();
            std::string str = args.args[0]->as_string().str();
            std::string prefix = args.args[1]->as_string().str();
            return mk_val<value_bool>(string_startswith(str, prefix));
        }},
        {"endswith", [](const func_args & args) -> value {
            args.ensure_vals<value_string, value_string>();
            std::string str = args.args[0]->as_string().str();
            std::string suffix = args.args[1]->as_string().str();
            return mk_val<value_bool>(string_endswith(str, suffix));
        }},
        {"split", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            std::string str = args.args[0]->as_string().str();
            std::string delim = (args.args.size() > 1) ? args.args[1]->as_string().str() : " ";
            auto result = mk_val<value_array>();
            size_t pos = 0;
            std::string token;
            while ((pos = str.find(delim)) != std::string::npos) {
                token = str.substr(0, pos);
                result->val_arr->push_back(mk_val<value_string>(token));
                str.erase(0, pos + delim.length());
            }
            auto res = mk_val<value_string>(str);
            res->val_str.mark_input_based_on(args.args[0]->val_str);
            result->val_arr->push_back(std::move(res));
            return std::move(result);
        }},
        {"replace", [](const func_args & args) -> value {
            args.ensure_vals<value_string, value_string, value_string>();
            std::string str = args.args[0]->as_string().str();
            std::string old_str = args.args[1]->as_string().str();
            std::string new_str = args.args[2]->as_string().str();
            size_t pos = 0;
            while ((pos = str.find(old_str, pos)) != std::string::npos) {
                str.replace(pos, old_str.length(), new_str);
                pos += new_str.length();
            }
            auto res = mk_val<value_string>(str);
            res->val_str.mark_input_based_on(args.args[0]->val_str);
            return res;
        }},
        {"int", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            std::string str = args.args[0]->as_string().str();
            try {
                return mk_val<value_int>(std::stoi(str));
            } catch (...) {
                throw std::runtime_error("Cannot convert string '" + str + "' to int");
            }
        }},
        {"float", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            std::string str = args.args[0]->as_string().str();
            try {
                return mk_val<value_float>(std::stod(str));
            } catch (...) {
                throw std::runtime_error("Cannot convert string '" + str + "' to float");
            }
        }},
        {"string", [](const func_args & args) -> value {
            // no-op
            args.ensure_vals<value_string>();
            return mk_val<value_string>(args.args[0]->as_string());
        }},
        {"indent", [](const func_args &) -> value {
            throw std::runtime_error("indent builtin not implemented");
        }},
        {"join", [](const func_args &) -> value {
            throw std::runtime_error("join builtin not implemented");
        }},
    };
    return builtins;
}


const func_builtins & value_bool_t::get_builtins() const {
    static const func_builtins builtins = {
        {"int", [](const func_args & args) -> value {
            args.ensure_vals<value_bool>();
            bool val = args.args[0]->as_bool();
            return mk_val<value_int>(val ? 1 : 0);
        }},
        {"float", [](const func_args & args) -> value {
            args.ensure_vals<value_bool>();
            bool val = args.args[0]->as_bool();
            return mk_val<value_float>(val ? 1.0 : 0.0);
        }},
        {"string", [](const func_args & args) -> value {
            args.ensure_vals<value_bool>();
            bool val = args.args[0]->as_bool();
            return mk_val<value_string>(val ? "True" : "False");
        }},
    };
    return builtins;
}


const func_builtins & value_array_t::get_builtins() const {
    static const func_builtins builtins = {
        {"list", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            const auto & arr = args.args[0]->as_array();
            auto result = mk_val<value_array>();
            for (const auto& v : arr) {
                result->val_arr->push_back(v->clone());
            }
            return result;
        }},
        {"first", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            const auto & arr = args.args[0]->as_array();
            if (arr.empty()) {
                return mk_val<value_undefined>();
            }
            return arr[0]->clone();
        }},
        {"last", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            const auto & arr = args.args[0]->as_array();
            if (arr.empty()) {
                return mk_val<value_undefined>();
            }
            return arr[arr.size() - 1]->clone();
        }},
        {"length", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            const auto & arr = args.args[0]->as_array();
            return mk_val<value_int>(static_cast<int64_t>(arr.size()));
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
            std::string key = args.args[1]->as_string().str();
            auto it = obj.find(key);
            if (it != obj.end()) {
                return it->second->clone();
            } else {
                return mk_val<value_undefined>();
            }
        }},
        {"keys", [](const func_args & args) -> value {
            args.ensure_vals<value_object>();
            const auto & obj = args.args[0]->as_object();
            auto result = mk_val<value_array>();
            for (const auto & pair : obj) {
                result->val_arr->push_back(mk_val<value_string>(pair.first));
            }
            return result;
        }},
        {"values", [](const func_args & args) -> value {
            args.ensure_vals<value_object>();
            const auto & obj = args.args[0]->as_object();
            auto result = mk_val<value_array>();
            for (const auto & pair : obj) {
                result->val_arr->push_back(pair.second->clone());
            }
            return result;
        }},
        {"items", [](const func_args & args) -> value {
            args.ensure_vals<value_object>();
            const auto & obj = args.args[0]->as_object();
            auto result = mk_val<value_array>();
            for (const auto & pair : obj) {
                auto item = mk_val<value_array>();
                item->val_arr->push_back(mk_val<value_string>(pair.first));
                item->val_arr->push_back(pair.second->clone());
                result->val_arr->push_back(std::move(item));
            }
            return result;
        }},
    };
    return builtins;
}

} // namespace jinja
