#include "jinja-lexer.h"
#include "jinja-vm.h"
#include "jinja-parser.h"
#include "jinja-value.h"

#include <string>
#include <cctype>
#include <vector>
#include <optional>
#include <algorithm>

namespace jinja {

/**
 * Function that mimics Python's array slicing.
 */
template<typename T>
static T slice(const T & array, std::optional<int64_t> start = std::nullopt, std::optional<int64_t> stop = std::nullopt, int64_t step = 1) {
    int64_t len = static_cast<int64_t>(array.size());
    int64_t direction = (step > 0) ? 1 : ((step < 0) ? -1 : 0);
    int64_t start_val;
    int64_t stop_val;
    if (direction >= 0) {
        start_val = start.value_or(0);
        if (start_val < 0) {
            start_val = std::max(len + start_val, (int64_t)0);
        } else {
            start_val = std::min(start_val, len);
        }

        stop_val = stop.value_or(len);
        if (stop_val < 0) {
            stop_val = std::max(len + stop_val, (int64_t)0);
        } else {
            stop_val = std::min(stop_val, len);
        }
    } else {
        start_val = start.value_or(len - 1);
        if (start_val < 0) {
            start_val = std::max(len + start_val, (int64_t)-1);
        } else {
            start_val = std::min(start_val, len - 1);
        }

        stop_val = stop.value_or(-1);
        if (stop_val < -1) {
            stop_val = std::max(len + stop_val, (int64_t)-1);
        } else {
            stop_val = std::min(stop_val, len - 1);
        }
    }
    T result;
    if (direction == 0) {
        return result;
    }
    for (int64_t i = start_val; direction * i < direction * stop_val; i += step) {
        if (i >= 0 && i < len) {
            result.push_back(std::move(array[static_cast<size_t>(i)]->clone()));
        }
    }
    return result;
}

template<typename T>
static value test_type_fn(const func_args & args) {
    args.ensure_count(1);
    bool is_type = is_val<T>(args.args[0]);
    return mk_val<value_bool>(is_type);
}
template<typename T, typename U>
static value test_type_fn(const func_args & args) {
    args.ensure_count(1);
    bool is_type = is_val<T>(args.args[0]) || is_val<U>(args.args[0]);
    return mk_val<value_bool>(is_type);
}

const func_builtins & global_builtins() {
    static const func_builtins builtins = {
        {"raise_exception", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            std::string msg = args.args[0]->as_string().str();
            throw raised_exception("Jinja Exception: " + msg);
        }},
        {"namespace", [](const func_args & args) -> value {
            auto out = mk_val<value_object>();
            for (const auto & arg : args.args) {
                if (!is_val<value_kwarg>(arg)) {
                    throw raised_exception("namespace() arguments must be kwargs");
                }
                auto kwarg = dynamic_cast<value_kwarg_t*>(arg.get());
                out->insert(kwarg->key, kwarg->val);
            }
            return out;
        }},

        // tests
        {"test_is_boolean", test_type_fn<value_bool>},
        {"test_is_callable", test_type_fn<value_func>},
        {"test_is_odd", [](const func_args & args) -> value {
            args.ensure_vals<value_int>();
            int64_t val = args.args[0]->as_int();
            return mk_val<value_bool>(val % 2 != 0);
        }},
        {"test_is_even", [](const func_args & args) -> value {
            args.ensure_vals<value_int>();
            int64_t val = args.args[0]->as_int();
            return mk_val<value_bool>(val % 2 == 0);
        }},
        {"test_is_false", [](const func_args & args) -> value {
            args.ensure_count(1);
            bool val = is_val<value_bool>(args.args[0]) && !args.args[0]->as_bool();
            return mk_val<value_bool>(val);
        }},
        {"test_is_true", [](const func_args & args) -> value {
            args.ensure_count(1);
            bool val = is_val<value_bool>(args.args[0]) && args.args[0]->as_bool();
            return mk_val<value_bool>(val);
        }},
        {"test_is_string", test_type_fn<value_string>},
        {"test_is_integer", test_type_fn<value_int>},
        {"test_is_number", test_type_fn<value_int, value_float>},
        {"test_is_iterable", test_type_fn<value_array, value_string>},
        {"test_is_mapping", test_type_fn<value_object>},
        {"test_is_lower", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            return mk_val<value_bool>(args.args[0]->val_str.is_lowercase());
        }},
        {"test_is_upper", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            return mk_val<value_bool>(args.args[0]->val_str.is_uppercase());
        }},
        {"test_is_none", test_type_fn<value_null>},
        {"test_is_defined", [](const func_args & args) -> value {
            args.ensure_count(1);
            return mk_val<value_bool>(!is_val<value_undefined>(args.args[0]));
        }},
        {"test_is_undefined", test_type_fn<value_undefined>},
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
        {"slice", [](const func_args &) -> value {
            throw std::runtime_error("slice builtin not implemented");
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
        {"slice", [](const func_args & args) -> value {
            args.ensure_count(4);
            int64_t start = is_val<value_int>(args.args[1]) ? args.args[1]->as_int() : 0;
            int64_t stop  = is_val<value_int>(args.args[2]) ? args.args[2]->as_int() : -1;
            int64_t step  = is_val<value_int>(args.args[3]) ? args.args[3]->as_int() : 1;
            if (!is_val<value_array>(args.args[0])) {
                throw raised_exception("slice() first argument must be an array");
            }
            if (step == 0) {
                throw raised_exception("slice step cannot be zero");
            }
            auto arr = slice(args.args[0]->as_array(), start, stop, step);
            auto res = mk_val<value_array>();
            res->val_arr = std::make_shared<std::vector<value>>(std::move(arr));
            return res;
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
