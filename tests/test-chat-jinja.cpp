#include <string>
#include <vector>
#include <sstream>
#include <regex>
#include <iostream>

#undef NDEBUG
#include <cassert>

#include "jinja/jinja-parser.h"
#include "jinja/jinja-lexer.h"

int main(void) {
    //std::string contents = "{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\\n' + message['content'] | trim + '<end_of_turn>\\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\\n'}}{% endif %}";

    std::string contents = "{{ ('hi' + 'fi') | upper }}";

    std::cout << "=== INPUT ===\n" << contents << "\n\n";

    jinja::lexer lexer;
    jinja::preprocess_options options;
    options.trim_blocks = true;
    options.lstrip_blocks = false;
    auto tokens = lexer.tokenize(contents, options);
    for (const auto & tok : tokens) {
        std::cout << "token: type=" << static_cast<int>(tok.t) << " text='" << tok.value << "'\n";
    }

    std::cout << "\n=== AST ===\n";
    jinja::program ast = jinja::parse_from_tokens(tokens);
    for (const auto & stmt : ast.body) {
        std::cout << "stmt type: " << stmt->type() << "\n";
    }

    std::cout << "\n=== OUTPUT ===\n";
    jinja::context ctx;
    jinja::vm vm(ctx);
    auto results = vm.execute(ast);
    for (const auto & res : results) {
        std::cout << "result type: " << res->type() << "\n";
        std::cout << "result value: " << res->as_string() << "\n";
    }

    return 0;
}
