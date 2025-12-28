#include <string>
#include <vector>
#include <sstream>
#include <regex>
#include <iostream>
#include <fstream>

#undef NDEBUG
#include <cassert>

#include "jinja/jinja-parser.h"
#include "jinja/jinja-lexer.h"

int main(void) {
    //std::string contents = "{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\\n' + message['content'] | trim + '<end_of_turn>\\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\\n'}}{% endif %}";

    //std::string contents = "{% if messages[0]['role'] != 'system' %}nice {{ messages[0]['content'] }}{% endif %}";

    //std::string contents = "<some_tokens> {{ messages[0]['content'] }} <another_token>";

    std::ifstream infile("models/templates/Qwen-Qwen3-0.6B.jinja");
    std::string contents((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());

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

    std::cout << "\n=== RUN ===\n";
    jinja::context ctx;

    auto make_non_special_string = [](const std::string & s) {
        jinja::value_string str_val = jinja::mk_val<jinja::value_string>(s);
        str_val->mark_input();
        return str_val;
    };

    jinja::value_array messages = jinja::mk_val<jinja::value_array>();
    jinja::value_object msg1 = jinja::mk_val<jinja::value_object>();
    msg1->insert("role",    make_non_special_string("user"));
    msg1->insert("content", make_non_special_string("Hello, how are you?"));
    messages->push_back(std::move(msg1));
    jinja::value_object msg2 = jinja::mk_val<jinja::value_object>();
    msg2->insert("role",    make_non_special_string("assistant"));
    msg2->insert("content", make_non_special_string("I am fine, thank you!"));
    messages->push_back(std::move(msg2));

    ctx.var["messages"] = std::move(messages);

    jinja::vm vm(ctx);
    const jinja::value results = vm.execute(ast);
    auto parts = vm.gather_string_parts(results);

    std::cout << "\n=== RESULTS ===\n";
    for (const auto & part : parts) {
        std::cout << (part.is_input ? "DATA" : "TMPL") << ": " << part.val << "\n";
    }

    return 0;
}
