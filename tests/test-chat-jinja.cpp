#include <string>
#include <vector>
#include <sstream>
#include <regex>
#include <iostream>
#include <fstream>
#include <filesystem>

#include <nlohmann/json.hpp>

#undef NDEBUG
#include <cassert>

#include "jinja/jinja-parser.h"
#include "jinja/jinja-lexer.h"
#include "jinja/jinja-type-infer.h"

void run_multiple();
void run_single(std::string contents);

int main(void) {
    //std::string contents = "{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\\n' + message['content'] | trim + '<end_of_turn>\\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\\n'}}{% endif %}";

    //std::string contents = "{% if messages[0]['role'] != 'system' %}nice {{ messages[0]['content'] }}{% endif %}";

    //std::string contents = "<some_tokens> {{ messages[a]['content'] }} <another_token>";
    //std::string contents = "{% if a is not defined %}hello{% endif %}";

    std::ifstream infile("models/templates/Qwen-Qwen3-0.6B.jinja");
    //std::ifstream infile("models/templates/Kimi-K2-Thinking.jinja");
    std::string contents((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());

    run_single(contents);

    //run_multiple();

    return 0;
}

void run_multiple(void) {
    std::vector<std::string> failed_tests;

    bool stop_on_first_failure = false;

    auto is_ignored_file = [](const std::string & filename) -> bool {
        std::vector<std::string> ignored_files = {
            "Apriel-",
            "Olmo-3-7B-Instruct-Heretic-GGUF",
            "sheldonrobinson-Llama-Guard",
            "deepseek-community-Janus-Pro-1B",
            "bitshrine-gemma-2-2B-function-calling",
            "PaddlePaddle-PaddleOCR-VL",
        };
        for (const auto & ignored : ignored_files) {
            if (filename.find(ignored) != std::string::npos) {
                return true;
            }
        }
        return false;
    };

    // list all files in models/templates/ and run each
    size_t test_count = 0;
    size_t skip_count = 0;
    //std::string dir_path = "models/templates/";
    std::string dir_path = "../test-jinja/templates/";
    for (const auto & entry : std::filesystem::directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            if (is_ignored_file(entry.path().filename().string())) {
                std::cout << "=== SKIPPING TEMPLATE FILE: " << entry.path().string() << " ===\n";
                skip_count++;
                continue;
            }

            test_count++;
            std::cout << "\n\n=== RUNNING TEMPLATE FILE: " << entry.path().string() << " ===\n";
            std::ifstream infile(entry.path());
            std::string contents((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());
            try {
                run_single(contents);
            } catch (const std::exception & e) {
                std::cout << "Exception: " << e.what() << "\n";
                std::cout << "=== ERROR WITH TEMPLATE FILE: " << entry.path().string() << " ===\n";
                failed_tests.push_back(entry.path().string());
                if (stop_on_first_failure) {
                    break;
                }
            }
        }
    }

    std::cout << "\n\n=== TEST SUMMARY ===\n";
    std::cout << "Total tests run: " << test_count << "\n";
    std::cout << "Total failed tests: " << failed_tests.size() << "\n";
    std::cout << "Total skipped tests: " << skip_count << "\n";
    for (const auto & test : failed_tests) {
        std::cout << "FAILED TEST: " << test << "\n";
    }
}


void run_single(std::string contents) {
    jinja::enable_debug(true);

    // lexing
    jinja::lexer lexer;
    jinja::preprocess_options options;
    options.trim_blocks = false;
    options.lstrip_blocks = false;
    auto lexer_res = lexer.tokenize(contents, options);

    // compile to AST
    jinja::program ast = jinja::parse_from_tokens(lexer_res);

    std::cout << "\n=== RUN ===\n";
    jinja::context ctx;
    ctx.source = lexer_res.preprocessed_source;

    std::string json_inp = R"({
        "messages": [
            {
                "role": "user",
                "content": {"__input__": "Hello, how are you?"}
            },
            {
                "role": "assistant",
                "content": {"__input__": "I am fine, thank you!"}
            },
            {
                "role": "assistant",
                "content": "Calling weather tool.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {
                                "location": "New York",
                                "unit": "celsius"
                            }
                        }
                    }
                ]
            }
        ],
        "bos_token": "<s>",
        "eos_token": "</s>",
        "tools": []
    })";
    auto input_json = nlohmann::json::parse(json_inp);

    // workaround for functionary models
    input_json["functions"] = "";
    input_json["datetime"] = "";

    // workaround for Llama Guard models
    input_json["excluded_category_keys"] = nlohmann::json::array();

    jinja::global_from_json(ctx, input_json);

    jinja::vm vm(ctx);
    const jinja::value results = vm.execute(ast);
    auto parts = vm.gather_string_parts(results);

    std::cout << "\n=== RESULTS ===\n";
    for (const auto & part : parts.get()->val_str.parts) {
        std::cout << (part.is_input ? "DATA" : "TMPL") << ": " << part.val << "\n";
    }

    std::cout << "\n=== TYPES ===\n";
    auto & global_obj = ctx.flatten_globals;
    for (const auto & pair : global_obj) {
        std::string name = pair.first;
        std::string inf_types;
        for (const auto & t : pair.second->inf_types) {
            inf_types += inferred_type_to_string(t) + " ";
        }
        if (inf_types.empty()) {
            continue;
        }
        std::string inf_vals;
        for (const auto & v : pair.second->inf_vals) {
            inf_vals += v->as_string().str() + " ; ";
        }
        printf("Var: %-20s | Types: %-10s | Vals: %s\n", name.c_str(), inf_types.c_str(), inf_vals.c_str());
    }
}
