#include "jinja-lexer.h"
#include "jinja-vm.h"
#include "jinja-parser.h"
#include "jinja-value.h"

#include <string>
#include <vector>
#include <memory>
#include <algorithm>

namespace jinja {

template<typename T>
static bool is_stmt(const statement_ptr & ptr) {
    return dynamic_cast<const T*>(ptr.get()) != nullptr;
}

value binary_expression::execute(context & ctx) {
    value left_val = left->execute(ctx);

    // Logical operators
    if (op.value == "and") {
        return left_val->as_bool() ? right->execute(ctx) : std::move(left_val);
    } else if (op.value == "or") {
        return left_val->as_bool() ? std::move(left_val) : right->execute(ctx);
    }

    // Equality operators
    value right_val = right->execute(ctx);
    if (op.value == "==") {
        return std::make_unique<value_bool_t>(left_val == right_val);
    } else if (op.value == "!=") {
        return std::make_unique<value_bool_t>(left_val != right_val);
    }

    // Handle undefined and null values
    if (is_val<value_undefined>(left_val) || is_val<value_undefined>(right_val)) {
        if (is_val<value_undefined>(right_val) && (op.value == "in" || op.value == "not in")) {
            // Special case: `anything in undefined` is `false` and `anything not in undefined` is `true`
            return std::make_unique<value_bool_t>(op.value == "not in");
        }
        throw std::runtime_error("Cannot perform operation " + op.value + " on undefined values");
    } else if (is_val<value_null>(left_val) || is_val<value_null>(right_val)) {
        throw std::runtime_error("Cannot perform operation on null values");
    }

    // String concatenation with ~
    if (op.value == "~") {
        return std::make_unique<value_string_t>(left_val->as_string() + right_val->as_string());
    }

    // Float operations
    if ((is_val<value_int>(left_val) || is_val<value_float>(left_val)) &&
        (is_val<value_int>(right_val) || is_val<value_float>(right_val))) {
        double a = left_val->as_float();
        double b = right_val->as_float();
        if (op.value == "+" || op.value == "-" || op.value == "*") {
            double res = (op.value == "+") ? a + b : (op.value == "-") ? a - b : a * b;
            bool is_float = is_val<value_float>(left_val) || is_val<value_float>(right_val);
            if (is_float) {
                return std::make_unique<value_float_t>(res);
            } else {
                return std::make_unique<value_int_t>(static_cast<int64_t>(res));
            }
        } else if (op.value == "/") {
            return std::make_unique<value_float_t>(a / b);
        } else if (op.value == "%") {
            double rem = std::fmod(a, b);
            bool is_float = is_val<value_float>(left_val) || is_val<value_float>(right_val);
            if (is_float) {
                return std::make_unique<value_float_t>(rem);
            } else {
                return std::make_unique<value_int_t>(static_cast<int64_t>(rem));
            }
        } else if (op.value == "<") {
            return std::make_unique<value_bool_t>(a < b);
        } else if (op.value == ">") {
            return std::make_unique<value_bool_t>(a > b);
        } else if (op.value == ">=") {
            return std::make_unique<value_bool_t>(a >= b);
        } else if (op.value == "<=") {
            return std::make_unique<value_bool_t>(a <= b);
        }
    }

    // Array operations
    if (is_val<value_array>(left_val) && is_val<value_array>(right_val)) {
        if (op.value == "+") {
            auto & left_arr = left_val->as_array();
            auto & right_arr = right_val->as_array();
            auto result = std::make_unique<value_array_t>();
            for (const auto & item : left_arr) {
                result->val_arr->push_back(item->clone());
            }
            for (const auto & item : right_arr) {
                result->val_arr->push_back(item->clone());
            }
            return result;
        }
    } else if (is_val<value_array>(right_val)) {
        auto & arr = right_val->as_array();
        bool member = std::find_if(arr.begin(), arr.end(), [&](const value& v) { return v == left_val; }) != arr.end();
        if (op.value == "in") {
            return std::make_unique<value_bool_t>(member);
        } else if (op.value == "not in") {
            return std::make_unique<value_bool_t>(!member);
        }
    }

    // String concatenation
    if (is_val<value_string>(left_val) || is_val<value_string>(right_val)) {
        if (op.value == "+") {
            return std::make_unique<value_string_t>(left_val->as_string() + right_val->as_string());
        }
    }

    // String membership
    if (is_val<value_string>(left_val) && is_val<value_string>(right_val)) {
        auto left_str = left_val->as_string();
        auto right_str = right_val->as_string();
        if (op.value == "in") {
            return std::make_unique<value_bool_t>(right_str.find(left_str) != std::string::npos);
        } else if (op.value == "not in") {
            return std::make_unique<value_bool_t>(right_str.find(left_str) == std::string::npos);
        }
    }

    // String in object
    if (is_val<value_string>(left_val) && is_val<value_object>(right_val)) {
        auto key = left_val->as_string();
        auto & obj = right_val->as_object();
        bool has_key = obj.find(key) != obj.end();
        if (op.value == "in") {
            return std::make_unique<value_bool_t>(has_key);
        } else if (op.value == "not in") {
            return std::make_unique<value_bool_t>(!has_key);
        }
    }

    throw std::runtime_error("Unknown operator \"" + op.value + "\" between " + left_val->type() + " and " + right_val->type());
}

value filter_expression::execute(context & ctx) {
    value input = operand->execute(ctx);
    // value filter_func = filter->execute(ctx);

    if (is_stmt<identifier>(filter)) {
        auto filter_val = dynamic_cast<identifier*>(filter.get())->value;

        if (filter_val == "to_json") {
            // TODO: Implement to_json filter
            throw std::runtime_error("to_json filter not implemented");
        }

        if (is_val<value_array>(input)) {
            auto & arr = input->as_array();
            if (filter_val == "list") {
                return std::make_unique<value_array_t>(input);
            } else if (filter_val == "first") {
                if (arr.empty()) {
                    return std::make_unique<value_undefined_t>();
                }
                return arr[0]->clone();
            } else if (filter_val == "last") {
                if (arr.empty()) {
                    return std::make_unique<value_undefined_t>();
                }
                return arr[arr.size() - 1]->clone();
            } else if (filter_val == "length") {
                return std::make_unique<value_int_t>(static_cast<int64_t>(arr.size()));
            } else {
                // TODO: reverse, sort, join, string, unique
                throw std::runtime_error("Unknown filter '" + filter_val + "' for array");
            }

        } else if (is_val<value_string>(input)) {
            auto str = input->as_string();
            auto builtins = input->get_builtins();
            auto it = builtins.find(filter_val);
            if (it != builtins.end()) {
                func_args args;
                args.args.push_back(input->clone());
                return it->second(args);
            }
            throw std::runtime_error("Unknown filter '" + filter_val + "' for string");

        } else if (is_val<value_int>(input) || is_val<value_float>(input)) {
            // TODO
            throw std::runtime_error("Unknown filter '" + filter_val + "' for number");

        } else {
            throw std::runtime_error("Filters not supported for type " + input->type());
        }
    }
}

} // namespace jinja
