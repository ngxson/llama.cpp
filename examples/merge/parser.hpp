#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <set>
#include <string.h>

// trim whitespace from the beginning and end of a string
static std::string str_trim(const std::string & str) {
    size_t start = 0;
    size_t end = str.size();
    while (start < end && isspace(str[start])) {
        start += 1;
    }
    while (end > start && isspace(str[end - 1])) {
        end -= 1;
    }
    return str.substr(start, end - start);
}

inline std::vector<std::string> str_split(std::string str, const std::string & delimiter) {
    size_t pos = 0;
    std::string token;
    std::vector<std::string> output;
    while ((pos = str.find(delimiter)) != std::string::npos) {
        token = str.substr(0, pos);
        output.push_back(token);
        str.erase(0, pos + delimiter.length());
    }
    output.push_back(str); // the rest
    return output;
}

/////////////////////////////////

// dump a list of tensor name of the input model
static std::vector<std::string> get_list_tensors_name(std::string & model_path) {
    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_load_model_from_file(model_path.c_str(), model_params);
    size_t n_tensors = llama_get_all_tensors_name(model, nullptr, 0);
    std::vector<const char *> list(n_tensors, nullptr);
    llama_get_all_tensors_name(model, list.data(), list.size());
    // copy the result
    std::vector<std::string> results;
    for (auto & name : list) {
        results.push_back(std::string(name));
    }
    llama_free_model(model);
    return results;
}

struct components_and_units {
    std::vector<std::string> tensors; // list of all tensor names
    std::set<std::string> components; // model components, for example "output", "token_embd",...
    std::set<std::string> units; // name of units inside layer, for example "attn_output"
};

// get layer index from tensor name, for example "blk.x.attn_norm.weight"
// returns -1 if it is non-layer
static int get_i_layer(std::string tensor_name) {
    int i_layer = -1;
    return sscanf(tensor_name.c_str(), "blk.%d.", &i_layer) == 1 ? i_layer : -1;
};

static struct components_and_units get_model_options(std::string & model_path) {
    struct components_and_units result;
    auto inp_names = get_list_tensors_name(model_path);
    std::set<std::string> units;
    for (auto & name : inp_names) {
        result.tensors.push_back(name);
        int il = get_i_layer(name);
        auto parts = str_split(name, ".");
        if (il < 0) {
            // non-layer components
            result.components.insert(parts[0]);
        } else {
            // tensor belong to layer
            result.units.insert(parts[2]);
        }
    }
    return result;
}

static void print_model_tensors_name(std::string & model_path) {
    auto opt = get_model_options(model_path);
    std::cout << "\n\n===================\n";
    std::cout << "Total number of tensors: " << opt.tensors.size() << "\n";
    for (size_t i = 0; i < opt.tensors.size(); i++) {
        char buf[128];
        sprintf(buf, "%4ld: %s", i, opt.tensors[i].c_str());
        std::cout << buf << "\n";
    }
    std::cout << "\n\n===================\n";
    std::cout << "\nComponents:\n";
    for (auto & c : opt.components) {
        std::cout << c << "\n";
    }
    std::cout << "\nList of layer units:\n";
    for (auto & u : opt.units) {
        std::cout << u << "\n";
    }
}

/////////////////////////////////

static void print_inst(struct llama_merge_inst inst) {
    std::cout << "Output: " << inst.name << "\n";
    switch (inst.method) {
        case LLAMA_MERGE_LINEAR:
            std::cout << "    Linear\n";
            std::cout << "    Model A: " << inst.scales[0] << " * " << inst.srcs[0] << "\n";
            std::cout << "    Model B: " << inst.scales[1] << " * " << inst.srcs[1] << "\n";
            break;
        case LLAMA_MERGE_SLERP:
            std::cout << "    SLERP t=" << inst.t << "\n";
            std::cout << "    Model A: " << inst.srcs[0] << "\n";
            std::cout << "    Model B: " << inst.srcs[1] << "\n";
            break;
        case LLAMA_MERGE_COPY:
            std::cout << "    Copy from model A: "<< inst.srcs[0] << "\n";
            break;
        case LLAMA_MERGE_REPEAT:
            std::cout << "    Repeat from output model: " << inst.srcs[0] << "\n";
            break;
        default:
            break;
    }
}

static std::vector<struct llama_merge_inst> parse_config(std::string & config_path, std::string & model_path, size_t & n_layers) {
    std::vector<struct llama_merge_inst> instructions;

    // read file
    std::ifstream file(config_path);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file merge config file");
    }
    std::ostringstream content;
    content << file.rdbuf(); // Read the entire file into the stringstream
    auto lines = str_split(content.str(), "\n");
    file.close();

    // get list of input tensors
    struct components_and_units opt = get_model_options(model_path);
    auto & units = opt.units;
    std::unordered_map<std::string, struct llama_merge_inst> comp; // map tensor name to instruction

    for (auto & c : opt.components) {
        struct llama_merge_inst ins;
        std::string name(c + ".weight");
        ins.method = LLAMA_MERGE_SLERP;
        strcpy(ins.name, name.c_str());
        strcpy(ins.srcs[0], name.c_str());
        strcpy(ins.srcs[1], name.c_str());
        ins.t = 0.5; // by default
        comp[c] = ins;
    }

    // process line by line, one line is one layer
    std::unordered_map<std::string, struct llama_merge_inst> layer; // map tensor name to instruction
    bool is_layer_empty = true;
    int i_layer = -1;
    auto get_tensor_name = [&](int layer, std::string unit) {
        return "blk." + std::to_string(layer) + "." + unit + ".weight";
    };
    auto push_output_layer = [&]() {
        if (!is_layer_empty) {
            for (auto & it : layer) {
                instructions.push_back(it.second);
            }
        }
        layer.clear();
        is_layer_empty = true;
    };
    auto new_output_layer = [&]() {
        layer.clear();
        for (auto & u : units) {
            struct llama_merge_inst ins;
            strcpy(ins.name, get_tensor_name(i_layer, u).c_str());
            layer[u] = ins;
        }
    };

    auto raise_err = [&](size_t i_line, std::string message) {
        std::stringstream ss;
        ss << "Parse error: (line " << i_line + 1 << ") " << message;
        throw std::runtime_error(ss.str());
    };

    // process "component" part
    for (size_t i_line = 0 ; i_line < lines.size(); i_line++) {
        auto line = str_trim(lines[i_line]);
        if (line.rfind("component", 0) != 0) {
            continue; // skip empty line not starting with "component"
        }

        auto parts = str_split(line, " ");
        auto verb = parts[1];
        auto params = str_split(parts[2], ",");
        struct llama_merge_inst ins;

        if (verb == "linear") {
            if (params.size() != 3) {
                raise_err(i_line, "[component] verb \"linear\" requires exactly 3 parameters");
            }
            ins.method = LLAMA_MERGE_LINEAR;
            std::string name(params[0] + ".weight");
            strcpy(ins.name, name.c_str());
            strcpy(ins.srcs[0], name.c_str());
            strcpy(ins.srcs[1], name.c_str());
            ins.scales[0] = std::stof(params[1]);
            ins.scales[1] = std::stof(params[2]);
            comp[params[0]] = ins;
        } else if (verb == "slerp") {
            if (params.size() != 2) {
                raise_err(i_line, "[component] verb \"slerp\" requires exactly 2 parameters");
            }
            ins.method = LLAMA_MERGE_SLERP;
            std::string name(params[0] + ".weight");
            strcpy(ins.name, name.c_str());
            strcpy(ins.srcs[0], name.c_str());
            strcpy(ins.srcs[1], name.c_str());
            ins.t = std::stof(params[1]);
            comp[params[0]] = ins;
        } else if (verb == "copy") {
            raise_err(i_line, "verb \"copy\" is not supported for components, please use \"linear\" instead");
        } else {
            raise_err(i_line, "invalid verb: " + verb);
        }
    }

    // push all components
    for (auto & c : comp) {
        instructions.push_back(c.second);
    }

    // process "layer" part
    for (size_t i_line = 0 ; i_line < lines.size(); i_line++) {
        auto line = str_trim(lines[i_line]);
        if (line.empty() || line.c_str()[0] == '#' || line.rfind("component", 0) == 0) {
            continue; // skip empty line, comment or component definition
        }

        auto parts = str_split(line, " ");
        if (parts.size() != 3) {
            raise_err(i_line, "does not follow format: \"target (space) verb (space) parameters\"");
        }

        auto target = parts[0];
        auto verb = parts[1];
        auto params = str_split(parts[2], ",");

        if (target == "output" && verb == "layer") {
            int il_curr = std::stoi(params[0]);
            if (i_layer + 1 != il_curr) {
                raise_err(i_line, "new layer number must be (last layer number + 1)");
            }
            push_output_layer();
            i_layer = il_curr;
            new_output_layer();
            continue;
        }

        auto linear = [&](struct llama_merge_inst & ins, std::string unit) {
            if (params.size() != 4) {
                raise_err(i_line, "verb \"linear\" requires exactly 4 parameters");
            }
            ins.method = LLAMA_MERGE_LINEAR;
            int src0 = std::stoi(params[0]);
            int src1 = std::stoi(params[1]);
            strcpy(ins.srcs[0], get_tensor_name(src0, unit).c_str());
            strcpy(ins.srcs[1], get_tensor_name(src1, unit).c_str());
            ins.scales[0] = std::stof(params[2]);
            ins.scales[1] = std::stof(params[3]);
            is_layer_empty = false;
        };

        auto slerp = [&](struct llama_merge_inst & ins, std::string unit) {
            if (params.size() != 3) {
                raise_err(i_line, "verb \"slerp\" requires exactly 3 parameters");
            }
            ins.method = LLAMA_MERGE_SLERP;
            int src0 = std::stoi(params[0]);
            int src1 = std::stoi(params[1]);
            strcpy(ins.srcs[0], get_tensor_name(src0, unit).c_str());
            strcpy(ins.srcs[1], get_tensor_name(src1, unit).c_str());
            ins.t = std::stof(params[2]);
            is_layer_empty = false;
        };

        /*auto repeat = [&](struct llama_merge_inst & ins, std::string unit) {
            if (params.size() != 1) {
                raise_err(i_line, "verb \"repeat\" requires exactly 1 parameter");
            }
            ins.method = LLAMA_MERGE_REPEAT;
            int src0 = std::stoi(params[0]);
            strcpy(ins.srcs[0], get_tensor_name(src0, unit).c_str());
            is_layer_empty = false;
        };*/

        auto copy = [&](struct llama_merge_inst & ins, std::string unit) {
            if (params.size() != 2) {
                raise_err(i_line, "verb \"copy\" requires exactly 2 parameters");
            }
            ins.method = LLAMA_MERGE_COPY;
            int model = std::stoi(params[0]);
            int layer = std::stoi(params[1]);
            if (model == 0) {
                strcpy(ins.srcs[0], get_tensor_name(layer, unit).c_str());
                strcpy(ins.srcs[1], "");
            } else if (model == 1) {
                strcpy(ins.srcs[0], "");
                strcpy(ins.srcs[1], get_tensor_name(layer, unit).c_str());
            } else {
                raise_err(i_line, "can only copy from model 0 or 1");
            }
            is_layer_empty = false;
        };

        auto apply_verb = [&](struct llama_merge_inst & ins, std::string unit) {
            if (verb == "linear") {
                linear(ins, unit);
            } else if (verb == "slerp") {
                slerp(ins, unit);
            } else if (verb == "repeat") {
                // repeat(ins, unit);
                raise_err(i_line, "repeat is currently not supported");
            } else if (verb == "copy") {
                copy(ins, unit);
            } else {
                raise_err(i_line, "invalid verb: " + verb);
            }
        };

        // TODO: what if user does not use "all"? we may miss some tensors?
        if (target == "all") {
            for (auto & u : units) {
                apply_verb(layer[u], u);
            }
        } else {
            if (units.find(target) == units.end()) {
                raise_err(i_line, "unit " + target + " does not exist");
            }
            apply_verb(layer[target], target);
        }
    }
    push_output_layer();
    n_layers = i_layer + 1;

    // print all parsed instructions
    std::cout << "Parsed instructions:\n";
    for (auto & ins : instructions) {
        print_inst(ins);
    }
    std::cout << "---\n" << "Total output layers: " << n_layers << "\n";

    return instructions;
}
