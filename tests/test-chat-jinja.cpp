#include <string>
#include <vector>
#include <sstream>
#include <regex>
#include <iostream>

#undef NDEBUG
#include <cassert>

#include "peg-parser.h"
#include "json-schema-to-grammar.h"
#include "jinja/jinja-compiler.h"
#include "jinja/jinja-lexer.h"

int main(void) {
    std::string contents = "{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\\n' + message['content'] | trim + '<end_of_turn>\\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\\n'}}{% endif %}";

    std::cout << "=== INPUT ===\n" << contents << "\n\n";

    jinja::lexer lexer;
    jinja::preprocess_options options;
    options.trim_blocks = true;
    options.lstrip_blocks = false;
    auto tokens = lexer.tokenize(contents, options);
    for (const auto & tok : tokens) {
        std::cout << "token: type=" << static_cast<int>(tok.t) << " text='" << tok.value << "'\n";
    }

    // jinja::compiler compiler;
    // compiler.builder.set_root(compiler.root);
    // auto parser = compiler.builder.build();

    // auto grammar = build_grammar([&](const common_grammar_builder & builder0) {
    //     parser.build_grammar(builder0);
    // });
    // printf("== GRAMMAR ==\n");
    // printf("%s\n", grammar.c_str());

    // // printf("== DUMP ==\n");
    // // printf("%s\n", parser.dump(compiler.root.id()).c_str());

    // printf("== PARSE ==\n");

    // common_peg_parse_context ctx(contents);
    // const auto result = parser.parse(ctx);
    // if (!result.success()) {
    //     throw std::runtime_error("failed to parse, type = " + std::to_string(result.type));
    // }

    // ctx.ast.visit(result, [&](const common_peg_ast_node & node) {
    //     printf("node: rule='%s' text='%s'\n", node.rule.c_str(), std::string(node.text).c_str());
    // });

    return 0;
}
