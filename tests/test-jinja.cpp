#include <string>
#include <iostream>

#include <nlohmann/json.hpp>

#include "jinja/jinja-interpreter.h"
#include "jinja/jinja-parser.h"
#include "jinja/jinja-lexer.h"

#include "testing.h"

using json = nlohmann::ordered_json;

static void assert_template(testing & t, const std::string & tmpl, const json & vars, const std::string & expect);

static void test_whitespace_control(testing & t) {
    t.test("no whitespace control", [](testing & t) {
        assert_template(t,
            "    {% if true %}\n    {% endif %}",
            json::object(),
            "    \n    "
        );

        assert_template(t,
            "  {% if kvs %}"
            "   {% for k, v in kvs %}{{ k }}={{ v }} {% endfor %}"
            "  {% endif %}",
            {{"kvs", {{"a", 1}, {"b", 2}}}},
            "     a=1 b=2   "
        );
    });

    t.test("leading whitespace control", [](testing & t) {
        assert_template(t,
            "  {%- if kvs %}"
            "   {%- for k, v in kvs %}{{ k }}={{ v }} {% endfor -%}"
            "  {%- endif %}",
            {{"kvs", {{"a", 1}, {"b", 2}}}},
            "a=1 b=2 "
        );

        assert_template(t,
            "{{- ']~b[ai' ~ '\\n' }}\n"
            "\n"
            "{%- set reasoning_content = ''%}",
            json::object(),
            "]~b[ai\n"
        );
    });
}

int main(int argc, char *argv[]) {
    testing t(std::cout);
    t.verbose = true;

    if (argc >= 2) {
        t.set_filter(argv[1]);
    }

    t.test("whitespace", test_whitespace_control);

    return t.summary();
}

static void assert_template(testing & t, const std::string & tmpl, const json & vars, const std::string & expect) {
    jinja::lexer lexer;
    auto lexer_res = lexer.tokenize(tmpl);

    jinja::program ast = jinja::parse_from_tokens(lexer_res);

    jinja::context ctx(tmpl);
    jinja::global_from_json(ctx, vars);

    jinja::interpreter interpreter(ctx);

    const jinja::value results = interpreter.execute(ast);
    auto parts = interpreter.gather_string_parts(results);

    std::string rendered;
    for (const auto & part : parts->as_string().parts) {
        rendered += part.val;
    }

    if (!t.assert_true("Template render mismatch", expect == rendered)) {
        t.log("Template: " + json(tmpl).dump());
        t.log("Expected: " + json(expect).dump());
        t.log("Actual  : " + json(rendered).dump());
    }
}
