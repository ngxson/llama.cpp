#pragma once

#include "server-common.h"
#include "server-task.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

// typed decision models (TypeSafe /v1/systemone API)
// the model answers each question in one forward pass, no token is generated

enum server_decision_type {
    SERVER_DECISION_TYPE_NONE,    // not a decision model
    SERVER_DECISION_TYPE_OPENJEV, // logits of one label token per option, read at the last prompt token
    SERVER_DECISION_TYPE_LAYA,    // score of one marker token per option, read from the embeddings output
};

enum server_decision_question_type {
    SERVER_DECISION_QUESTION_CHOICE,
    SERVER_DECISION_QUESTION_SCORE,
    SERVER_DECISION_QUESTION_NOUL,
};

struct server_decision_option {
    std::string key;
    json description; // null if not provided
};

struct server_decision_question {
    std::string id;
    server_decision_question_type type;
    json instructions;
    std::vector<server_decision_option> options; // in the order of the model outputs
};

struct server_decision_context {
    server_decision_type type = SERVER_DECISION_TYPE_NONE;

    // read the "<arch>.decision.*" metadata, type stays NONE if the model has none
    void init(const llama_model * model);

    // true if the result is read from the embeddings of each token
    bool need_embd() const { return type == SERVER_DECISION_TYPE_LAYA; }

    // throw std::invalid_argument on bad input
    std::vector<server_decision_question> parse_questions(const json & body) const;

    // set the prompt of this question, and where to read its result
    void fill_task(const json & state, const server_decision_question & question, server_task & task) const;

    // scores: one raw model output per option
    json format_answer(const server_decision_question & question, const std::vector<float> & scores) const;

private:
    const llama_vocab * vocab = nullptr;
    std::shared_ptr<const common_chat_template> tmpl; // the "systemone" template

    std::map<std::string, float> temperatures; // "<type>" or "<type>.<n_options bucket>"
    size_t n_options_max   = 0;
    bool   noul_true_first = false; // noul options are [true, false] instead of [false, true]

    // OPENJEV
    std::vector<llama_token> labels;

    // LAYA
    llama_token token_marker      = LLAMA_TOKEN_NULL;
    llama_token token_sep         = LLAMA_TOKEN_NULL;
    std::string text_marker;
    size_t      max_head_tokens   = 0; // question + options
    size_t      max_option_tokens = 48;

    std::string render(const json & state, const server_decision_question & question) const;
    void fill_task_laya(llama_tokens & tokens, const server_decision_question & question, server_task & task) const;

    float get_temperature(const server_decision_question & question) const;
};
