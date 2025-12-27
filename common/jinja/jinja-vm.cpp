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
        return mk_val<value_bool>(left_val == right_val);
    } else if (op.value == "!=") {
        return mk_val<value_bool>(left_val != right_val);
    }

    // Handle undefined and null values
    if (is_val<value_undefined>(left_val) || is_val<value_undefined>(right_val)) {
        if (is_val<value_undefined>(right_val) && (op.value == "in" || op.value == "not in")) {
            // Special case: `anything in undefined` is `false` and `anything not in undefined` is `true`
            return mk_val<value_bool>(op.value == "not in");
        }
        throw std::runtime_error("Cannot perform operation " + op.value + " on undefined values");
    } else if (is_val<value_null>(left_val) || is_val<value_null>(right_val)) {
        throw std::runtime_error("Cannot perform operation on null values");
    }

    // String concatenation with ~
    if (op.value == "~") {
        return mk_val<value_string>(left_val->as_string() + right_val->as_string());
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
                return mk_val<value_float>(res);
            } else {
                return mk_val<value_int>(static_cast<int64_t>(res));
            }
        } else if (op.value == "/") {
            return mk_val<value_float>(a / b);
        } else if (op.value == "%") {
            double rem = std::fmod(a, b);
            bool is_float = is_val<value_float>(left_val) || is_val<value_float>(right_val);
            if (is_float) {
                return mk_val<value_float>(rem);
            } else {
                return mk_val<value_int>(static_cast<int64_t>(rem));
            }
        } else if (op.value == "<") {
            return mk_val<value_bool>(a < b);
        } else if (op.value == ">") {
            return mk_val<value_bool>(a > b);
        } else if (op.value == ">=") {
            return mk_val<value_bool>(a >= b);
        } else if (op.value == "<=") {
            return mk_val<value_bool>(a <= b);
        }
    }

    // Array operations
    if (is_val<value_array>(left_val) && is_val<value_array>(right_val)) {
        if (op.value == "+") {
            auto & left_arr = left_val->as_array();
            auto & right_arr = right_val->as_array();
            auto result = mk_val<value_array>();
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
            return mk_val<value_bool>(member);
        } else if (op.value == "not in") {
            return mk_val<value_bool>(!member);
        }
    }

    // String concatenation
    if (is_val<value_string>(left_val) || is_val<value_string>(right_val)) {
        if (op.value == "+") {
            return mk_val<value_string>(left_val->as_string() + right_val->as_string());
        }
    }

    // String membership
    if (is_val<value_string>(left_val) && is_val<value_string>(right_val)) {
        auto left_str = left_val->as_string();
        auto right_str = right_val->as_string();
        if (op.value == "in") {
            return mk_val<value_bool>(right_str.find(left_str) != std::string::npos);
        } else if (op.value == "not in") {
            return mk_val<value_bool>(right_str.find(left_str) == std::string::npos);
        }
    }

    // String in object
    if (is_val<value_string>(left_val) && is_val<value_object>(right_val)) {
        auto key = left_val->as_string();
        auto & obj = right_val->as_object();
        bool has_key = obj.find(key) != obj.end();
        if (op.value == "in") {
            return mk_val<value_bool>(has_key);
        } else if (op.value == "not in") {
            return mk_val<value_bool>(!has_key);
        }
    }

    throw std::runtime_error("Unknown operator \"" + op.value + "\" between " + left_val->type() + " and " + right_val->type());
}

value filter_expression::execute(context & ctx) {
    value input = operand->execute(ctx);

    auto try_builtin = [&](const std::string & name) -> value {
        auto builtins = input->get_builtins();
        auto it = builtins.find(name);
        if (it != builtins.end()) {
            func_args args;
            args.args.push_back(input->clone());
            return it->second(args);
        }
        return nullptr;
    };

    if (is_stmt<identifier>(filter)) {
        auto filter_val = dynamic_cast<identifier*>(filter.get())->value;

        if (filter_val == "to_json") {
            // TODO: Implement to_json filter
            throw std::runtime_error("to_json filter not implemented");
        }

        if (is_val<value_array>(input)) {
            auto & arr = input->as_array();
            auto res = try_builtin(filter_val);
            if (res) {
                return res;
            }
            throw std::runtime_error("Unknown filter '" + filter_val + "' for array");

        } else if (is_val<value_string>(input)) {
            auto str = input->as_string();
            auto builtins = input->get_builtins();
            if (filter_val == "trim") {
                filter_val = "strip"; // alias
            }
            auto res = try_builtin(filter_val);
            if (res) {
                return res;
            }
            throw std::runtime_error("Unknown filter '" + filter_val + "' for string");

        } else if (is_val<value_int>(input) || is_val<value_float>(input)) {
            auto res = try_builtin(filter_val);
            if (res) {
                return res;
            }
            throw std::runtime_error("Unknown filter '" + filter_val + "' for number");

        } else {
            throw std::runtime_error("Filters not supported for type " + input->type());
        }

    } else if (is_stmt<call_expression>(filter)) {
        // TODO
        // value filter_func = filter->execute(ctx);
        throw std::runtime_error("Filter with arguments not implemented");

    } else {
        throw std::runtime_error("Invalid filter expression");
    }
}

value if_statement::execute(context & ctx) {
    throw std::runtime_error("if_statement::execute not implemented");
}

value for_statement::execute(context & ctx) {
    throw std::runtime_error("for_statement::execute not implemented");
}

value break_statement::execute(context & ctx) {
    throw std::runtime_error("break_statement::execute not implemented");
}

value continue_statement::execute(context & ctx) {
    throw std::runtime_error("continue_statement::execute not implemented");
}

value set_statement::execute(context & ctx) {
    throw std::runtime_error("set_statement::execute not implemented");
}

} // namespace jinja
