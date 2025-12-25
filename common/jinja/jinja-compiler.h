#include "common.h"
#include <chat-peg-parser.h>
#include <sstream>

namespace jinja {

struct compiler {
    common_chat_peg_native_builder builder;
    common_peg_parser root;

    compiler() : root(builder.choice()) {
        auto & p = builder;

        auto ws = p.rule("ws", p.chars("[ \t]", 0, -1));
        auto num = p.rule("num", p.chars("[0-9]", 1, -1));

        //
        // expressions
        //

        auto expression = p.choice();

        auto var_name = p.rule("var_name", p.chars("[a-zA-Z_]", 1, -1) << p.chars("[a-zA-Z0-9_]", 0, -1));
        expression |= var_name;

        // value
        auto p_int = p.rule("value_int", num);
        auto p_flt = p.rule("value_flt", num << "." << p.optional(num));
        auto p_str = p.rule("value_str",
            p.json_string() |
            p.literal("'") + p.chars("[^']*", 0, -1) + p.literal("'")
        );

        expression |= p_int;
        expression |= p_flt;
        expression |= p_str;

        // function calls
        auto p_args = p.rule("args", expression << ws << p.zero_or_more("," << ws << expression));
        auto p_func = p.rule("func", ws << var_name << ws << "(" << ws << p_args << ws << ")");
        expression |= p_func;

        // indexing
        auto p_idx = p.rule("idx", ws << "[" << ws << expression << ws << "]");
        expression |= p_idx;

        // set
        auto p_set = p.rule("set", "set " << ws << var_name << ws << "=" << expression);
        expression |= p_set;

        // if, else, endif
        auto p_if    = p.rule("if", "if " << ws << expression << ws);
        auto p_else  = p.rule("else", "else " << ws << expression << ws);
        auto p_endif = p.rule("endif", p.literal("endif"));

        expression |= p_if;
        expression |= p_else;
        expression |= p_endif;

        expression = p.space() + expression + p.space();

        //
        // root
        //

        // auto strip = p.rule("strip", "-" << expression << "-");
        auto print = p.rule("print", "{{" << (expression) << "}}");
        auto ctrl  = p.rule("ctrl",  "{%" << (expression) << "%}");

        root |= print;
        root |= ctrl;
        root |= p.rule("text", p.negate(root));

        root = p.one_or_more(root);
        root += p.end();
    }
};

} // namespace jinja
