#pragma once

#include <vector>

#include "jinja-value.h"
#include "jinja-vm.h"

#define FILENAME "jinja-caps"

namespace jinja {

struct caps {
    bool content_string = true;
    bool content_array = true;
};

using caps_messages_fn = std::function<value()>;
using caps_analyze_fn = std::function<void(bool, value &, value &)>;
static void caps_try_execute(jinja::program & prog,
                                caps_messages_fn messages_fn,
                                caps_messages_fn tools_fn,
                                caps_analyze_fn analyze_fn) {
    context ctx;
    ctx.is_get_stats = true;

    value messages = messages_fn();
    value tools = tools_fn();

    ctx.set_val("messages", messages);
    ctx.set_val("tools", tools);
    ctx.set_val("add_generation_prompt", mk_val<value_bool>(true));

    bool success = false;
    try {
        jinja::vm vm(ctx);
        vm.execute(prog);
        success = true;
    } catch (const std::exception & e) {
        JJ_DEBUG("Exception during execution: %s", e.what());
        // ignore exceptions during capability analysis
    }
    return analyze_fn(success, messages, tools);
}

// for debugging only
static void caps_print_stats(value & v, std::string path) {
    std::string ops;
    for (const auto & name : v->stats.ops) {
        ops += name + " ";
    }
    JJ_DEBUG("Value %s, type: %s %s, ops: %s",
                path.c_str(),
                v->type().c_str(),
                v->stats.used ? "(used)" : "",
                ops.c_str());
}

static caps caps_get(jinja::program & prog) {
    caps result;

    static const auto has_op = [](value & v, const std::string & op_name) {
        return v->stats.ops.find(op_name) != v->stats.ops.end();
    };

    // case: given content as string, check if it's accessed as array
    caps_try_execute(
        prog,
        [&]() {
            auto messages = mk_val<value_array>();
            {
                value_object msg = mk_val<value_object>();
                msg->insert("role", mk_val<value_string>("user"));
                msg->insert("content", mk_val<value_string>("User message"));
                messages->push_back(msg);
            }
            return messages;
        },
        [&]() {
            return mk_val<value_array>();
        },
        [&](bool, value & messages, value &) {
            auto & content = messages->at(0)->at("content");
            caps_print_stats(content, "messages[0].content");
            if (has_op(content, "selectattr") || has_op(content, "array_access")) {
                // accessed as an array
                JJ_DEBUG("%s", "Force content as array");
                result.content_string = false;
                result.content_array = true;
            }
        }
    );

    // case: given content as array, check if it's supported or not
    caps_try_execute(
        prog,
        [&]() {
            auto messages = mk_val<value_array>();
            {
                value_object msg = mk_val<value_object>();
                msg->insert("role", mk_val<value_string>("user"));
                value_array content_arr = mk_val<value_array>();
                {
                    value_object content_part = mk_val<value_object>();
                    content_part->insert("type", mk_val<value_string>("text"));
                    content_part->insert("text", mk_val<value_string>("User message"));
                    content_arr->push_back(content_part);
                }
                msg->insert("content", content_arr);
                messages->push_back(msg);
            }
            return messages;
        },
        [&]() {
            return mk_val<value_array>();
        },
        [&](bool success, value & messages, value &) {
            auto & content = messages->at(0)->at("content");
            caps_print_stats(content, "messages[0].content");
            if (!success) {
                JJ_DEBUG("%s", "Cannot handle content as array");
                result.content_array = false;
            }
        }
    );

    return result;
}

static void caps_apply_workarounds(context & ctx, const caps & c) {
    auto messages = ctx.get_val("messages");

    if (!is_val<value_array>(messages)) {
        throw std::runtime_error("Expected messages to be an array");
    }

    if (!c.content_string) {
        for (auto & msg : messages->val_arr) {
            if (!is_val<value_object>(msg)) {
                throw std::runtime_error("Expected messages[i] to be an object");
            }
            auto obj_ptr = cast_val<value_object>(msg);
            auto & content = obj_ptr->at("content");
            if (!is_val<value_array>(content)) {
                JJ_DEBUG("%s", "Converting message content to array");
                auto str_content = content->as_string();
                value_array arr_content = mk_val<value_array>();
                value_object content_part = mk_val<value_object>();
                content_part->insert("type", mk_val<value_string>("text"));
                content_part->insert("text", mk_val<value_string>(str_content));
                arr_content->push_back(content_part);
                obj_ptr->insert("content", arr_content);
            }
        }
    }

    ctx.set_val("messages", messages);

    //
    // per-model workarounds
    //

    // workaround for shieldgemma-2b-Q2_K
    if (ctx.get_val("guideline")->is_undefined()) {
        ctx.set_val("guideline", mk_val<value_string>(""));
    }

    // workaround for functionary models
    if (ctx.get_val("functions")->is_undefined()) {
        ctx.set_val("functions", mk_val<value_string>(""));
    }
    if (ctx.get_val("datetime")->is_undefined()) {
        ctx.set_val("datetime", mk_val<value_string>(""));
    }

    // workaround for Llama-3-5B-Sheard
    if (ctx.get_val("system_message")->is_undefined()) {
        ctx.set_val("system_message", mk_val<value_string>(""));
    }
}

} // namespace jinja
