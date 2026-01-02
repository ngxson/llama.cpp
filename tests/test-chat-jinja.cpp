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

using json = nlohmann::json;

void run_multiple(std::string dir_path, bool stop_on_first_failure, json input);
void run_single(std::string contents, json input, const std::string & output_path = "");

std::string HELP = R"(
Usage: test-chat-jinja [OPTIONS] PATH_TO_TEMPLATE
Options:
  -h, --help               Show this help message and exit.
  --json <path>            Path to the JSON input file.
  --stop-on-first-fail     Stop testing on the first failure (default: false).
  --output <path>          Path to output results (only for single template runs).
If PATH_TO_TEMPLATE is a file, runs that single template.
If PATH_TO_TEMPLATE is a directory, runs all .jinja files in that directory.
)";

std::string DEFAULT_JSON = R"({
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
    "tools": [],
    "add_generation_prompt": true
})";

int main(int argc, char ** argv) {
    std::vector<std::string> args(argv, argv + argc);

    std::string tmpl_path;
    std::string json_path;
    std::string output_path;
    bool stop_on_first_fail = false;

    for (size_t i = 1; i < args.size(); i++) {
        if (args[i] == "--help" || args[i] == "-h") {
            std::cout << HELP << "\n";
            return 0;
        } else if (args[i] == "--json" && i + 1 < args.size()) {
            json_path = args[i + 1];
            i++;
        } else if (args[i] == "--stop-on-first-fail") {
            stop_on_first_fail = true;
        } else if (args[i] == "--output" && i + 1 < args.size()) {
            output_path = args[i + 1];
            i++;
        } else if (tmpl_path.empty()) {
            tmpl_path = args[i];
        } else {
            std::cerr << "Unknown argument: " << args[i] << "\n";
            std::cout << HELP << "\n";
            return 1;
        }
    }

    if (tmpl_path.empty()) {
        std::cerr << "Error: PATH_TO_TEMPLATE is required.\n";
        std::cout << HELP << "\n";
        return 1;
    }

    json input_json;
    if (!json_path.empty()) {
        std::ifstream json_file(json_path);
        if (!json_file) {
            std::cerr << "Error: Could not open JSON file: " << json_path << "\n";
            return 1;
        }
        std::string content = std::string(
            std::istreambuf_iterator<char>(json_file),
            std::istreambuf_iterator<char>());
        input_json = json::parse(content);
    } else {
        input_json = json::parse(DEFAULT_JSON);
    }

    std::filesystem::path p(tmpl_path);
    if (std::filesystem::is_directory(p)) {
        run_multiple(tmpl_path, stop_on_first_fail, input_json);
    } else if (std::filesystem::is_regular_file(p)) {
        std::ifstream infile(tmpl_path);
        std::string contents = std::string(
            std::istreambuf_iterator<char>(infile),
            std::istreambuf_iterator<char>());
        run_single(contents, input_json, output_path);
    } else {
        std::cerr << "Error: PATH_TO_TEMPLATE is not a valid file or directory: " << tmpl_path << "\n";
        return 1;
    }

    return 0;
}

void run_multiple(std::string dir_path, bool stop_on_first_fail, json input) {
    std::vector<std::string> failed_tests;

    // list all files in models/templates/ and run each
    size_t test_count = 0;

    for (const auto & entry : std::filesystem::directory_iterator(dir_path)) {
        // only process .jinja files
        if (entry.path().extension() == ".jinja" && entry.is_regular_file()) {
            test_count++;
            std::cout << "\n\n=== RUNNING TEMPLATE FILE: " << entry.path().string() << " ===\n";
            std::ifstream infile(entry.path());
            std::string contents((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());
            try {
                run_single(contents, input);
            } catch (const std::exception & e) {
                std::cout << "Exception: " << e.what() << "\n";
                std::cout << "=== ERROR WITH TEMPLATE FILE: " << entry.path().string() << " ===\n";
                failed_tests.push_back(entry.path().string());
                if (stop_on_first_fail) {
                    break;
                }
            }
        }
    }

    std::cout << "\n\n=== TEST SUMMARY ===\n";
    std::cout << "Total tests run: " << test_count << "\n";
    std::cout << "Total failed tests: " << failed_tests.size() << "\n";
    for (const auto & test : failed_tests) {
        std::cout << "FAILED TEST: " << test << "\n";
    }
}


void run_single(std::string contents, json input, const std::string & output_path) {
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

    jinja::global_from_json(ctx, input);

    jinja::vm vm(ctx);
    const jinja::value results = vm.execute(ast);
    auto parts = vm.gather_string_parts(results);

    std::cout << "\n=== RESULTS ===\n";
    for (const auto & part : parts->as_string().parts) {
        std::cout << (part.is_input ? "DATA" : "TMPL") << ": " << part.val << "\n";
    }

    if (!output_path.empty()) {
        std::ofstream outfile(output_path);
        if (!outfile) {
            throw std::runtime_error("Could not open output file: " + output_path);
        }
        for (const auto & part : parts->as_string().parts) {
            outfile << part.val;
        }
        std::cout << "\n=== OUTPUT WRITTEN TO " << output_path << " ===\n";
    }
}
