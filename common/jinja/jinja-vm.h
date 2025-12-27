#pragma once

#include "jinja-lexer.h"
#include "jinja-value.h"

#include <string>
#include <vector>
#include <cassert>
#include <memory>
#include <sstream>


namespace jinja {

struct context {
    std::map<std::string, value> var;

    context() = default;
    ~context() = default;

    context(const context & parent) {
        // inherit variables (for example, when entering a new scope)
        for (const auto & pair : parent.var) {
            var[pair.first] = pair.second->clone();
        }
    }
};

/**
 * Base class for all nodes in the AST.
 */
struct statement {
    virtual ~statement() = default;
    virtual std::string type() const { return "Statement"; }
    virtual value execute(context & ctx) { throw std::runtime_error("cannot exec " + type()); }
};

using statement_ptr = std::unique_ptr<statement>;
using statements = std::vector<statement_ptr>;

// Type Checking Utilities

template<typename T>
static void chk_type(const statement_ptr & ptr) {
    if (!ptr) return; // Allow null for optional fields
    assert(dynamic_cast<T *>(ptr.get()) != nullptr);
}

template<typename T, typename U>
static void chk_type(const statement_ptr & ptr) {
    if (!ptr) return;
    assert(dynamic_cast<T *>(ptr.get()) != nullptr || dynamic_cast<U *>(ptr.get()) != nullptr);
}

// Base Types

/**
 * Expressions will result in a value at runtime (unlike statements).
 */
struct expression : public statement {
    std::string type() const override { return "Expression"; }
};

// Statements

struct program : public statement {
    statements body;

    explicit program(statements && body) : body(std::move(body)) {}
    std::string type() const override { return "Program"; }
    value execute(context & ctx) override {
        throw std::runtime_error("Cannot execute program directly, use jinja::vm instead");
    }
};

struct if_statement : public statement {
    statement_ptr test;
    statements body;
    statements alternate;

    if_statement(statement_ptr && test, statements && body, statements && alternate)
        : test(std::move(test)), body(std::move(body)), alternate(std::move(alternate)) {
        chk_type<expression>(this->test);
    }

    std::string type() const override { return "If"; }
    value execute(context & ctx) override;
};

struct identifier;
struct tuple_literal;

/**
 * Loop over each item in a sequence
 * https://jinja.palletsprojects.com/en/3.0.x/templates/#for
 */
struct for_statement : public statement {
    statement_ptr loopvar; // Identifier | TupleLiteral
    statement_ptr iterable;
    statements body;
    statements default_block; // if no iteration took place

    for_statement(statement_ptr && loopvar, statement_ptr && iterable, statements && body, statements && default_block)
        : loopvar(std::move(loopvar)), iterable(std::move(iterable)), 
          body(std::move(body)), default_block(std::move(default_block)) {
        chk_type<identifier, tuple_literal>(this->loopvar);
        chk_type<expression>(this->iterable);
    }

    std::string type() const override { return "For"; }
    value execute(context & ctx) override;
};

struct break_statement : public statement {
    std::string type() const override { return "Break"; }
    value execute(context & ctx) override;
};

struct continue_statement : public statement {
    std::string type() const override { return "Continue"; }
    value execute(context & ctx) override;
};

struct set_statement : public statement {
    statement_ptr assignee;
    statement_ptr val;
    statements body;

    set_statement(statement_ptr && assignee, statement_ptr && value, statements && body)
        : assignee(std::move(assignee)), val(std::move(value)), body(std::move(body)) {
        chk_type<expression>(this->assignee);
        chk_type<expression>(this->val);
    }

    std::string type() const override { return "Set"; }
    value execute(context & ctx) override;
};

struct macro_statement : public statement {
    statement_ptr name;
    statements args;
    statements body;

    macro_statement(statement_ptr && name, statements && args, statements && body)
        : name(std::move(name)), args(std::move(args)), body(std::move(body)) {
        chk_type<identifier>(this->name);
        for (const auto& arg : this->args) chk_type<expression>(arg);
    }

    std::string type() const override { return "Macro"; }
    value execute(context & ctx) override {}
};

struct comment_statement : public statement {
    std::string val;
    explicit comment_statement(const std::string & v) : val(v) {}
    std::string type() const override { return "Comment"; }
    value execute(context & ctx) override {}
};

// Expressions

struct member_expression : public expression {
    statement_ptr object;
    statement_ptr property;
    bool computed;

    member_expression(statement_ptr && object, statement_ptr && property, bool computed)
        : object(std::move(object)), property(std::move(property)), computed(computed) {
        chk_type<expression>(this->object);
        chk_type<expression>(this->property);
    }
    std::string type() const override { return "MemberExpression"; }
    value execute(context & ctx) override;
};

struct call_expression : public expression {
    statement_ptr callee;
    statements args;

    call_expression(statement_ptr && callee, statements && args)
        : callee(std::move(callee)), args(std::move(args)) {
        chk_type<expression>(this->callee);
        for (const auto& arg : this->args) chk_type<expression>(arg);
    }
    std::string type() const override { return "CallExpression"; }
};

/**
 * Represents a user-defined variable or symbol in the template.
 */
struct identifier : public expression {
    std::string val;
    explicit identifier(const std::string & val) : val(val) {}
    std::string type() const override { return "Identifier"; }
    value execute(context & ctx) override;
};

// Literals

struct integer_literal : public expression { 
    int64_t val;
    explicit integer_literal(int64_t val) : val(val) {}
    std::string type() const override { return "IntegerLiteral"; }
    value execute(context & ctx) override {
        return std::make_unique<value_int_t>(val);
    }
};

struct float_literal : public expression {
    double val;
    explicit float_literal(double val) : val(val) {}
    std::string type() const override { return "FloatLiteral"; }
    value execute(context & ctx) override {
        return std::make_unique<value_float_t>(val);
    }
};

struct string_literal : public expression {
    std::string val;
    explicit string_literal(const std::string & val) : val(val) {}
    std::string type() const override { return "StringLiteral"; }
    value execute(context & ctx) override {
        return std::make_unique<value_string_t>(val);
    }
};

struct array_literal : public expression {
    statements val;
    explicit array_literal(statements && val) : val(std::move(val)) {
        for (const auto& item : this->val) chk_type<expression>(item);
    }
    std::string type() const override { return "ArrayLiteral"; }
};

struct tuple_literal : public expression {
    statements val;
    explicit tuple_literal(statements && val) : val(std::move(val)) {
        for (const auto & item : this->val) chk_type<expression>(item);
    }
    std::string type() const override { return "TupleLiteral"; }
};

struct object_literal : public expression {
    std::vector<std::pair<statement_ptr, statement_ptr>> val;
    explicit object_literal(std::vector<std::pair<statement_ptr, statement_ptr>> && val) 
        : val(std::move(val)) {
        for (const auto & pair : this->val) {
            chk_type<expression>(pair.first);
            chk_type<expression>(pair.second);
        }
    }
    std::string type() const override { return "ObjectLiteral"; }
};

// Complex Expressions

/**
 * An operation with two sides, separated by an operator.
 * Note: Either side can be a Complex Expression, with order
 * of operations being determined by the operator.
 */
struct binary_expression : public expression {
    token op;
    statement_ptr left;
    statement_ptr right;

    binary_expression(token op, statement_ptr && left, statement_ptr && right)
        : op(op), left(std::move(left)), right(std::move(right)) {
        chk_type<expression>(this->left);
        chk_type<expression>(this->right);
    }
    std::string type() const override { return "BinaryExpression"; }
    value execute(context & ctx) override;
};

/**
 * An operation with two sides, separated by the | operator.
 * Operator precedence: https://github.com/pallets/jinja/issues/379#issuecomment-168076202
 */
struct filter_expression : public expression {
    statement_ptr operand;
    statement_ptr filter;

    filter_expression(statement_ptr && operand, statement_ptr && filter)
        : operand(std::move(operand)), filter(std::move(filter)) {
        chk_type<expression>(this->operand);
        chk_type<identifier, call_expression>(this->filter);
    }
    std::string type() const override { return "FilterExpression"; }
    value execute(context & ctx) override;
};

struct filter_statement : public statement {
    statement_ptr filter;
    statements body;

    filter_statement(statement_ptr && filter, statements && body)
        : filter(std::move(filter)), body(std::move(body)) {
        chk_type<identifier, call_expression>(this->filter);
    }
    std::string type() const override { return "FilterStatement"; }
    value execute(context & ctx) override {}
};

/**
 * An operation which filters a sequence of objects by applying a test to each object,
 * and only selecting the objects with the test succeeding.
 *
 * It may also be used as a shortcut for a ternary operator.
 */
struct select_expression : public expression {
    statement_ptr lhs;
    statement_ptr test;

    select_expression(statement_ptr && lhs, statement_ptr && test)
        : lhs(std::move(lhs)), test(std::move(test)) {
        chk_type<expression>(this->lhs);
        chk_type<expression>(this->test);
    }
    std::string type() const override { return "SelectExpression"; }
};

/**
 * An operation with two sides, separated by the "is" operator.
 */
struct test_expression : public expression {
    statement_ptr operand;
    bool negate;
    statement_ptr test;

    test_expression(statement_ptr && operand, bool negate, statement_ptr && test)
        : operand(std::move(operand)), negate(negate), test(std::move(test)) {
        chk_type<expression>(this->operand);
        chk_type<identifier>(this->test);
    }
    std::string type() const override { return "TestExpression"; }
};

/**
 * An operation with one side (operator on the left).
 */
struct unary_expression : public expression {
    token op;
    statement_ptr argument;

    unary_expression(token op, statement_ptr && argument)
        : op(std::move(op)), argument(std::move(argument)) {
        chk_type<expression>(this->argument);
    }
    std::string type() const override { return "UnaryExpression"; }
};

struct slice_expression : public expression {
    statement_ptr start;
    statement_ptr stop;
    statement_ptr step;

    slice_expression(statement_ptr && start, statement_ptr && stop, statement_ptr && step)
        : start(std::move(start)), stop(std::move(stop)), step(std::move(step)) {
        chk_type<expression>(this->start);
        chk_type<expression>(this->stop);
        chk_type<expression>(this->step);
    }
    std::string type() const override { return "SliceExpression"; }
};

struct keyword_argument_expression : public expression {
    statement_ptr key;
    statement_ptr value;

    keyword_argument_expression(statement_ptr && key, statement_ptr && value)
        : key(std::move(key)), value(std::move(value)) {
        chk_type<identifier>(this->key);
        chk_type<expression>(this->value);
    }
    std::string type() const override { return "KeywordArgumentExpression"; }
};

struct spread_expression : public expression {
    statement_ptr argument;
    explicit spread_expression(statement_ptr && argument) : argument(std::move(argument)) {
        chk_type<expression>(this->argument);
    }
    std::string type() const override { return "SpreadExpression"; }
};

struct call_statement : public statement {
    statement_ptr call;
    statements caller_args;
    statements body;

    call_statement(statement_ptr && call, statements && caller_args, statements && body)
        : call(std::move(call)), caller_args(std::move(caller_args)), body(std::move(body)) {
        chk_type<call_expression>(this->call);
        for (const auto& arg : this->caller_args) chk_type<expression>(arg);
    }
    std::string type() const override { return "CallStatement"; }
    value execute(context & ctx) override {}
};

struct ternary_expression : public expression {
    statement_ptr condition;
    statement_ptr true_expr;
    statement_ptr false_expr;

    ternary_expression(statement_ptr && condition, statement_ptr && true_expr, statement_ptr && false_expr)
        : condition(std::move(condition)), true_expr(std::move(true_expr)), false_expr(std::move(false_expr)) {
        chk_type<expression>(this->condition);
        chk_type<expression>(this->true_expr);
        chk_type<expression>(this->false_expr);
    }
    std::string type() const override { return "Ternary"; }
};

//////////////////////

struct vm {
    context & ctx;
    explicit vm(context & ctx) : ctx(ctx) {}

    std::vector<value> execute(program & prog) {
        std::vector<value> results;
        for (auto & stmt : prog.body) {
            value res = stmt->execute(ctx);
            results.push_back(std::move(res));
        }
        return results;
    }
};

} // namespace jinja
