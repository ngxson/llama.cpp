#include "jinja-lexer.h"
#include "jinja-vm.h"
#include "jinja-parser.h"
#include "jinja-value.h"

#include <string>
#include <vector>
#include <memory>
#include <algorithm>

#define JJ_DEBUG(msg, ...)  printf("jinja-vm:%3d : " msg "\n", __LINE__, __VA_ARGS__)
//#define JJ_DEBUG(msg, ...)  // no-op

namespace jinja {

template<typename T>
static bool is_stmt(const statement_ptr & ptr) {
    return dynamic_cast<const T*>(ptr.get()) != nullptr;
}

static value_array exec_statements(const statements & stmts, context & ctx) {
    auto result = mk_val<value_array>();
    for (const auto & stmt : stmts) {
        JJ_DEBUG("Executing statement of type %s", stmt->type().c_str());
        result->val_arr->push_back(stmt->execute(ctx));
    }
    return result;
}

value identifier::execute(context & ctx) {
    auto it = ctx.var.find(val);
    auto builtins = global_builtins();
    if (it != ctx.var.end()) {
        JJ_DEBUG("Identifier '%s' found", val.c_str());
        return it->second->clone();
    } else if (builtins.find(val) != builtins.end()) {
        JJ_DEBUG("Identifier '%s' found in builtins", val.c_str());
        return mk_val<value_func>(builtins.at(val));
    } else {
        JJ_DEBUG("Identifier '%s' not found, returning undefined", val.c_str());
        return mk_val<value_undefined>();
    }
}

value binary_expression::execute(context & ctx) {
    value left_val = left->execute(ctx);
    JJ_DEBUG("Executing binary expression %s '%s' %s", left_val->type().c_str(), op.value.c_str(), right->type().c_str());

    // Logical operators
    if (op.value == "and") {
        return left_val->as_bool() ? right->execute(ctx) : std::move(left_val);
    } else if (op.value == "or") {
        return left_val->as_bool() ? std::move(left_val) : right->execute(ctx);
    }

    // Equality operators
    value right_val = right->execute(ctx);
    if (op.value == "==") {
        return mk_val<value_bool>(value_compare(left_val, right_val));
    } else if (op.value == "!=") {
        return mk_val<value_bool>(!value_compare(left_val, right_val));
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

    // Float operations
    if ((is_val<value_int>(left_val) || is_val<value_float>(left_val)) &&
        (is_val<value_int>(right_val) || is_val<value_float>(right_val))) {
        double a = left_val->as_float();
        double b = right_val->as_float();
        if (op.value == "+" || op.value == "-" || op.value == "*") {
            double res = (op.value == "+") ? a + b : (op.value == "-") ? a - b : a * b;
            JJ_DEBUG("Arithmetic operation: %f %s %f = %f", a, op.value.c_str(), b, res);
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
            JJ_DEBUG("Modulo operation: %f %% %f = %f", a, b, rem);
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

    // String concatenation with ~ and +
    if ((is_val<value_string>(left_val) || is_val<value_string>(right_val)) &&
            (op.value == "~" || op.value == "+")) {
        JJ_DEBUG("String concatenation with %s operator", op.value.c_str());
        auto output = left_val->as_string().append(right_val->as_string());
        auto res = mk_val<value_string>();
        res->val_str = std::move(output);
        return res;
    }

    // String membership
    if (is_val<value_string>(left_val) && is_val<value_string>(right_val)) {
        auto left_str = left_val->as_string().str();
        auto right_str = right_val->as_string().str();
        if (op.value == "in") {
            return mk_val<value_bool>(right_str.find(left_str) != std::string::npos);
        } else if (op.value == "not in") {
            return mk_val<value_bool>(right_str.find(left_str) == std::string::npos);
        }
    }

    // String in object
    if (is_val<value_string>(left_val) && is_val<value_object>(right_val)) {
        auto key = left_val->as_string().str();
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

static value try_builtin_func(const std::string & name, const value & input) {
    auto builtins = input->get_builtins();
    auto it = builtins.find(name);
    if (it != builtins.end()) {
        JJ_DEBUG("Binding built-in '%s'", name.c_str());
        return mk_val<value_func>(it->second, input, name);
    }
    throw std::runtime_error("Unknown (built-in) filter '" + name + "' for type " + input->type());
}

value filter_expression::execute(context & ctx) {
    value input = operand->execute(ctx);

    if (is_stmt<identifier>(filter)) {
        auto filter_val = dynamic_cast<identifier*>(filter.get())->val;

        if (filter_val == "to_json") {
            // TODO: Implement to_json filter
            throw std::runtime_error("to_json filter not implemented");
        }

        auto str = input->as_string();
        if (filter_val == "trim") {
            filter_val = "strip"; // alias
        }
        JJ_DEBUG("Applying filter '%s' to %s", filter_val.c_str(), input->type().c_str());
        return try_builtin_func(filter_val, input);

    } else if (is_stmt<call_expression>(filter)) {
        // TODO
        // value filter_func = filter->execute(ctx);
        throw std::runtime_error("Filter with arguments not implemented");

    } else {
        throw std::runtime_error("Invalid filter expression");
    }
}

value test_expression::execute(context & ctx) {
    // NOTE: "value is something" translates to function call "test_is_something(value)"
    const auto & builtins = global_builtins();
    if (!is_stmt<identifier>(test)) {
        throw std::runtime_error("Invalid test expression");
    }

    auto test_id = dynamic_cast<identifier*>(test.get())->val;
    auto it = builtins.find("test_is_" + test_id);
    JJ_DEBUG("Test expression %s '%s'", operand->type().c_str(), test_id.c_str());
    if (it == builtins.end()) {
        throw std::runtime_error("Unknown test '" + test_id + "'");
    }

    func_args args;
    args.args.push_back(operand->execute(ctx));
    return it->second(args);
}

value unary_expression::execute(context & ctx) {
    value operand_val = argument->execute(ctx);
    JJ_DEBUG("Executing unary expression with operator '%s'", op.value.c_str());

    if (op.value == "not") {
        return mk_val<value_bool>(!operand_val->as_bool());
    } else if (op.value == "-") {
        if (is_val<value_int>(operand_val)) {
            return mk_val<value_int>(-operand_val->as_int());
        } else if (is_val<value_float>(operand_val)) {
            return mk_val<value_float>(-operand_val->as_float());
        } else {
            throw std::runtime_error("Unary - operator requires numeric operand");
        }
    }

    throw std::runtime_error("Unknown unary operator '" + op.value + "'");
}

value if_statement::execute(context & ctx) {
    value test_val = test->execute(ctx);
    auto out = mk_val<value_array>();
    if (test_val->as_bool()) {
        for (auto & stmt : body) {
            JJ_DEBUG("IF --> Executing THEN body, current block: %s", stmt->type().c_str());
            out->val_arr->push_back(stmt->execute(ctx));
        }
    } else {
        for (auto & stmt : alternate) {
            JJ_DEBUG("IF --> Executing ELSE body, current block: %s", stmt->type().c_str());
            out->val_arr->push_back(stmt->execute(ctx));
        }
    }
    return out;
}

value for_statement::execute(context & ctx) {
    context scope(ctx); // new scope for loop variables

    statement_ptr iter_expr = std::move(iterable);
    statement_ptr test_expr = nullptr;

    if (is_stmt<select_expression>(iterable)) {
        JJ_DEBUG("%s", "For loop has test expression");
        auto select = dynamic_cast<select_expression*>(iterable.get());
        iter_expr = std::move(select->lhs);
        test_expr = std::move(select->test);
    }

    JJ_DEBUG("Executing for statement, iterable type: %s", iter_expr->type().c_str());

    value iterable_val = iter_expr->execute(scope);
    if (!is_val<value_array>(iterable_val) && !is_val<value_object>(iterable_val)) {
        throw std::runtime_error("Expected iterable or object type in for loop: got " + iterable_val->type());
    }

    std::vector<value> items;
    if (is_val<value_object>(iterable_val)) {
        auto & obj = iterable_val->as_object();
        for (auto & p : obj) {
            items.push_back(mk_val<value_string>(p.first));
        }
    } else {
        auto & arr = iterable_val->as_array();
        for (const auto & item : arr) {
            items.push_back(item->clone());
        }
    }

    std::vector<std::function<void(context &)>> scope_update_fns;

    std::vector<value> filtered_items;
    for (size_t i = 0; i < items.size(); ++i) {
        context loop_scope(scope);

        const value & current = items[i];

        std::function<void(context&)> scope_update_fn = [](context &) { /* no-op */};
        if (is_stmt<identifier>(loopvar)) {
            auto id = dynamic_cast<identifier*>(loopvar.get())->val;
            scope_update_fn = [id, &items, i](context & ctx) {
                ctx.var[id] = items[i]->clone();
            };
        } else if (is_stmt<tuple_literal>(loopvar)) {
            auto tuple = dynamic_cast<tuple_literal*>(loopvar.get());
            if (!is_val<value_array>(current)) {
                throw std::runtime_error("Cannot unpack non-iterable type: " + current->type());
            }
            auto & c_arr = current->as_array();
            if (tuple->val.size() != c_arr.size()) {
                throw std::runtime_error(std::string("Too ") + (tuple->val.size() > c_arr.size() ? "few" : "many") + " items to unpack");
            }
            scope_update_fn = [tuple, &items, i](context & ctx) {
                auto & c_arr = items[i]->as_array();
                for (size_t j = 0; j < tuple->val.size(); ++j) {
                    if (!is_stmt<identifier>(tuple->val[j])) {
                        throw std::runtime_error("Cannot unpack non-identifier type: " + tuple->val[j]->type());
                    }
                    auto id = dynamic_cast<identifier*>(tuple->val[j].get())->val;
                    ctx.var[id] = c_arr[j]->clone();
                }
            };
        } else {
            throw std::runtime_error("Invalid loop variable(s): " + loopvar->type());
        }
        if (test_expr) {
            scope_update_fn(loop_scope);
            value test_val = test_expr->execute(loop_scope);
            if (!test_val->as_bool()) {
                continue;
            }
        }
        filtered_items.push_back(current->clone());
        scope_update_fns.push_back(scope_update_fn);
    }
    
    auto result = mk_val<value_array>();

    bool noIteration = true;
    for (size_t i = 0; i < filtered_items.size(); ++i) {
        JJ_DEBUG("For loop iteration %zu/%zu", i + 1, filtered_items.size());
        value_object loop_obj = mk_val<value_object>();
        loop_obj->insert("index", mk_val<value_int>(i + 1));
        loop_obj->insert("index0", mk_val<value_int>(i));
        loop_obj->insert("revindex", mk_val<value_int>(filtered_items.size() - i));
        loop_obj->insert("revindex0", mk_val<value_int>(filtered_items.size() - i - 1));
        loop_obj->insert("first", mk_val<value_bool>(i == 0));
        loop_obj->insert("last", mk_val<value_bool>(i == filtered_items.size() - 1));
        loop_obj->insert("length", mk_val<value_int>(filtered_items.size()));
        loop_obj->insert("previtem", i > 0 ? filtered_items[i - 1]->clone() : mk_val<value_undefined>());
        loop_obj->insert("nextitem", i < filtered_items.size() - 1 ? filtered_items[i + 1]->clone() : mk_val<value_undefined>());
        ctx.var["loop"] = loop_obj->clone();
        scope_update_fns[i](ctx);
        try {
            for (auto & stmt : body) {
                value val = stmt->execute(ctx);
                result->push_back(val);
            }
        } catch (const continue_statement::exception &) {
            continue;
        } catch (const break_statement::exception &) {
            break;
        }
        noIteration = false;
    }
    if (noIteration) {
        for (auto & stmt : default_block) {
            value val = stmt->execute(ctx);
            result->push_back(val);
        }
    }

    return result;
}

value set_statement::execute(context & ctx) {
    auto rhs = val ? val->execute(ctx) : exec_statements(body, ctx);

    if (is_stmt<identifier>(assignee)) {
        auto var_name = dynamic_cast<identifier*>(assignee.get())->val;
        JJ_DEBUG("Setting variable '%s'", var_name.c_str());
        ctx.var[var_name] = rhs->clone();

    } else if (is_stmt<tuple_literal>(assignee)) {
        auto tuple = dynamic_cast<tuple_literal*>(assignee.get());
        if (!is_val<value_array>(rhs)) {
            throw std::runtime_error("Cannot unpack non-iterable type in set: " + rhs->type());
        }
        auto & arr = rhs->as_array();
        if (arr.size() != tuple->val.size()) {
            throw std::runtime_error(std::string("Too ") + (tuple->val.size() > arr.size() ? "few" : "many") + " items to unpack in set");
        }
        for (size_t i = 0; i < tuple->val.size(); ++i) {
            auto & elem = tuple->val[i];
            if (!is_stmt<identifier>(elem)) {
                throw std::runtime_error("Cannot unpack to non-identifier in set: " + elem->type());
            }
            auto var_name = dynamic_cast<identifier*>(elem.get())->val;
            ctx.var[var_name] = arr[i]->clone();
        }

    } else if (is_stmt<member_expression>(assignee)) {
        auto member = dynamic_cast<member_expression*>(assignee.get());
        value object = member->object->execute(ctx);
        if (!is_val<value_object>(object)) {
            throw std::runtime_error("Cannot assign to member of non-object");
        }
        if (member->computed) {
            throw std::runtime_error("Cannot assign to computed member");
        }
        if (!is_stmt<identifier>(member->property)) {
            throw std::runtime_error("Cannot assign to member with non-identifier property");
        }
        auto prop_name = dynamic_cast<identifier*>(member->property.get())->val;
        auto obj_ptr = dynamic_cast<value_object*>(object.get());
        JJ_DEBUG("Setting object property '%s'", prop_name.c_str());
        obj_ptr->get()->insert(prop_name, rhs->clone());

    } else {
        throw std::runtime_error("Invalid LHS inside assignment expression: " + assignee->type());
    }
    return mk_val<value_null>();
}

value macro_statement::execute(context & ctx) {
    std::string name = dynamic_cast<identifier*>(this->name.get())->val;
    const func_handler func = [this, &ctx, name](const func_args & args) -> value {
        JJ_DEBUG("Invoking macro '%s' with %zu arguments", name.c_str(), args.args.size());
        context macro_ctx(ctx); // new scope for macro execution

        // bind parameters
        size_t param_count = this->args.size();
        size_t arg_count = args.args.size();
        for (size_t i = 0; i < param_count; ++i) {
            std::string param_name = dynamic_cast<identifier*>(this->args[i].get())->val;
            if (i < arg_count) {
                macro_ctx.var[param_name] = args.args[i]->clone();
            } else {
                macro_ctx.var[param_name] = mk_val<value_undefined>();
            }
        }

        // execute macro body
        return exec_statements(this->body, macro_ctx);
    };

    JJ_DEBUG("Defining macro '%s' with %zu parameters", name.c_str(), args.size());
    ctx.var[name] = mk_val<value_func>(func);
    return mk_val<value_null>();
}

value member_expression::execute(context & ctx) {
    value object = this->object->execute(ctx);

    value property;
    if (this->computed) {
        JJ_DEBUG("Member expression, computing property type %s", this->property->type().c_str());
        property = this->property->execute(ctx);
    } else {
        property = mk_val<value_string>(dynamic_cast<identifier*>(this->property.get())->val);
    }

    JJ_DEBUG("Member expression on object type %s, property type %s", object->type().c_str(), property->type().c_str());

    value val = mk_val<value_undefined>();

    if (is_val<value_object>(object)) {
        if (!is_val<value_string>(property)) {
            throw std::runtime_error("Cannot access object with non-string: got " + property->type());
        }
        auto key = property->as_string().str();
        JJ_DEBUG("Accessing object property '%s'", key.c_str());
        auto & obj = object->as_object();
        auto it = obj.find(key);
        if (it != obj.end()) {
            val = it->second->clone();
        } else {
            val = try_builtin_func(key, object);
        }

    } else if (is_val<value_array>(object) || is_val<value_string>(object)) {
        if (is_val<value_int>(property)) {
            int64_t index = property->as_int();
            JJ_DEBUG("Accessing %s index %lld", is_val<value_array>(object) ? "array" : "string", index);
            if (is_val<value_array>(object)) {
                auto & arr = object->as_array();
                if (index >= 0 && index < static_cast<int64_t>(arr.size())) {
                    val = arr[index]->clone();
                }
            } else { // value_string
                auto str = object->as_string().str();
                if (index >= 0 && index < static_cast<int64_t>(str.size())) {
                    val = mk_val<value_string>(std::string(1, str[index]));
                }
            }
        } else if (is_val<value_string>(property)) {
            auto key = property->as_string().str();
            JJ_DEBUG("Accessing %s built-in '%s'", is_val<value_array>(object) ? "array" : "string", key.c_str());
            val = try_builtin_func(key, object);
        } else {
            throw std::runtime_error("Cannot access property with non-string/non-number: got " + property->type());
        }

    } else {
        if (!is_val<value_string>(property)) {
            throw std::runtime_error("Cannot access property with non-string: got " + property->type());
        }
        auto key = property->as_string().str();
        val = try_builtin_func(key, object);
    }

    return val;
}

static func_args gather_call_args(const statements & arg_stmts, context & ctx) {
    func_args args;
    for (auto & arg_stmt : arg_stmts) {
        args.args.push_back(arg_stmt->execute(ctx));
    }
    return args;
}

value call_expression::execute(context & ctx) {
    auto args = gather_call_args(this->args, ctx);
    value callee_val = callee->execute(ctx);
    JJ_DEBUG("Calling function of type %s with %zu arguments", callee_val->type().c_str(), args.args.size());
    if (!is_val<value_t>(callee_val)) {
        throw std::runtime_error("Callee is not a function: got " + callee_val->type());
    }
    return callee_val->invoke(args);
}

// compare operator for value_t
bool value_compare(const value & a, const value & b) {
    JJ_DEBUG("Comparing types: %s and %s", a->type().c_str(), b->type().c_str());
    // compare numeric types
    if ((is_val<value_int>(a) || is_val<value_float>(a)) &&
        (is_val<value_int>(b) || is_val<value_float>(b))){
        try {
            return a->as_float() == b->as_float();
        } catch (...) {}
    }
    // compare string and number
    // TODO: not sure if this is the right behavior
    if ((is_val<value_string>(b) && (is_val<value_int>(a) || is_val<value_float>(a))) ||
        (is_val<value_string>(a) && (is_val<value_int>(b) || is_val<value_float>(b)))) {
        try {
            return a->as_string().str() == b->as_string().str();
        } catch (...) {}
    }
    // compare boolean simple
    if (is_val<value_bool>(a) && is_val<value_bool>(b)) {
        return a->as_bool() == b->as_bool();
    }
    // compare string simple
    if (is_val<value_string>(a) && is_val<value_string>(b)) {
        return a->as_string().str() == b->as_string().str();
    }
    // compare by type
    if (a->type() != b->type()) {
        return false;
    }
    return false;
}

} // namespace jinja
