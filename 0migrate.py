from __future__ import annotations

from dataclasses import dataclass, field
import re

# IMPORTANT: REMOVE THIS FILE BEFORE MERGING THE PR

# this script is partially vibe-coded

MODEL_H = "src/llama-model.h"
MODEL_CPP = "src/llama-model.cpp"

MARKER_START_BUILD_GRAPH = "MARKER_START_MIGRATION_BUILD_GRAPH"
MARKER_END_BUILD_GRAPH = "MARKER_END_MIGRATION_BUILD_GRAPH"
MARKER_START_LOAD_HPARAMS = "MARKER_START_MIGRATION_LOAD_HPARAMS"
MARKER_END_LOAD_HPARAMS = "MARKER_END_MIGRATION_LOAD_HPARAMS"
MARKER_START_LOAD_TENSORS = "MARKER_START_MIGRATION_LOAD_TENSORS"
MARKER_END_LOAD_TENSORS = "MARKER_END_MIGRATION_LOAD_TENSORS"

ARCH_RE = re.compile(r"case\s+(LLM_ARCH_[A-Z0-9_]+)\s*:")
BUILD_RE = re.compile(r"llm_build_[a-z0-9_]+")
STRUCT_RE = re.compile(r"struct\s+(llm_build_[a-z0-9_]+)\b")


@dataclass
class ModelInfo:
  LLM_ARCH: str
  llm_builds: set[str] = field(default_factory=set)
  llm_build_name: str = ""
  code_graph: str = ""
  code_hparams: str = ""
  code_tensors: str = ""
  model_header: str = ""
  # transformed code
  new_struct_name: str = ""


with open(MODEL_H, "r") as f:
  model_h_content = f.read()

with open(MODEL_CPP, "r") as f:
  model_cpp_content = f.read()


def extract_marked_region(content: str, start_marker: str, end_marker: str) -> str:
  start = content.find(start_marker)
  if start == -1:
    raise ValueError(f"could not find start marker: {start_marker}")

  end = content.find(end_marker, start)
  if end == -1:
    raise ValueError(f"could not find end marker: {end_marker}")

  return content[start:end]


def count_braces(line: str) -> int:
  return line.count("{") - line.count("}")


def extract_struct_definitions(content: str) -> dict[str, str]:
  lines = content.splitlines()
  definitions: dict[str, str] = {}

  pending_template_lines: list[str] = []
  i = 0

  while i < len(lines):
    line = lines[i]
    stripped = line.strip()

    if stripped.startswith("template "):
      pending_template_lines.append(line)
      i += 1
      continue

    match = STRUCT_RE.search(line)
    if not match:
      pending_template_lines = []
      i += 1
      continue

    struct_name = match.group(1)
    block_lines = [*pending_template_lines, line]
    pending_template_lines = []

    brace_depth = count_braces(line)
    i += 1

    while i < len(lines):
      block_lines.append(lines[i])
      brace_depth += count_braces(lines[i])
      if brace_depth == 0 and lines[i].strip().endswith("};"):
        break
      i += 1

    definitions[struct_name] = "\n".join(block_lines)
    i += 1

  return definitions


def parse_switch_case_blocks(content: str, start_marker: str, end_marker: str) -> dict[str, str]:
  region = extract_marked_region(content, start_marker, end_marker)
  lines = region.splitlines()

  mapping: dict[str, str] = {}
  current_arches: list[str] = []
  current_handler_lines: list[str] = []

  switch_depth = 0
  inside_switch = False

  def finalize_current_handler() -> None:
    nonlocal current_arches, current_handler_lines
    if not current_arches:
      return

    code = "\n".join(current_handler_lines).rstrip()

    for arch in current_arches:
      mapping[arch] = code

    current_arches = []
    current_handler_lines = []

  for line in lines:
    stripped = line.strip()
    top_level_case = inside_switch and switch_depth == 1 and stripped.startswith("case ")
    top_level_default = inside_switch and switch_depth == 1 and stripped.startswith("default:")

    if top_level_case:
      if current_handler_lines:
        finalize_current_handler()

      arch_match = ARCH_RE.match(stripped)
      if arch_match:
        current_arches.append(arch_match.group(1))

    elif top_level_default:
      finalize_current_handler()

    elif current_arches:
      current_handler_lines.append(line)

    switch_depth += count_braces(line)
    if "switch (arch) {" in line:
      inside_switch = True

  finalize_current_handler()
  return mapping


def parse_build_graph_mapping(content: str) -> dict[str, ModelInfo]:
  blocks = parse_switch_case_blocks(content, MARKER_START_BUILD_GRAPH, MARKER_END_BUILD_GRAPH)
  mapping: dict[str, ModelInfo] = {}

  for arch, code_graph in blocks.items():
    mapping[arch] = ModelInfo(
      LLM_ARCH=arch,
      llm_builds=set(BUILD_RE.findall(code_graph)),
      code_graph=code_graph,
    )

  return mapping


def assign_code_blocks(mapping: dict[str, ModelInfo], blocks: dict[str, str], attr_name: str) -> None:
  for arch, code in blocks.items():
    if arch not in mapping:
      mapping[arch] = ModelInfo(LLM_ARCH=arch)

    setattr(mapping[arch], attr_name, code)


mapping = parse_build_graph_mapping(model_cpp_content)

assign_code_blocks(
  mapping,
  parse_switch_case_blocks(model_cpp_content, MARKER_START_LOAD_HPARAMS, MARKER_END_LOAD_HPARAMS),
  "code_hparams",
)

assign_code_blocks(
  mapping,
  parse_switch_case_blocks(model_cpp_content, MARKER_START_LOAD_TENSORS, MARKER_END_LOAD_TENSORS),
  "code_tensors",
)

for arch, info in mapping.items():
  if len(info.llm_builds) != 1:
    print(f"warning: expected exactly one llm_build for {arch}, found: {info.llm_builds}")
    continue

  info.llm_build_name = next(iter(info.llm_builds))
  print(f"{arch} -> {info.llm_build_name}")
  info.new_struct_name = info.llm_build_name.replace("llm_build_", "llama_model_")





output_select_arch_fn = ""
output_select_arch_fn += "switch (arch) {\n"
for arch, info in mapping.items():
  if not info.llm_build_name:
    continue

  output_select_arch_fn += "        case {}:\n".format(arch)
  output_select_arch_fn += "            {\n"
  output_select_arch_fn += "                model = new {}(params);\n".format(info.new_struct_name)
  output_select_arch_fn += "            } break;\n"
output_select_arch_fn += "        default:\n"
output_select_arch_fn += "            GGML_ABORT(\"unsupported architecture\");\n"
output_select_arch_fn += "    }\n"

print("\n\nSELECT_ARCH_FN:\n")
print(output_select_arch_fn)

model_cpp_content = model_cpp_content.replace("// SELECT_ARCH_FN", output_select_arch_fn)
with open(MODEL_CPP + ".log", "w") as f:
  f.write(model_cpp_content)






MODELS_H = "src/models/models.h"
with open(MODELS_H, "r") as f:
  models_h_content = f.read()

model_headers = extract_struct_definitions(models_h_content)

for arch, info in mapping.items():
  if not info.llm_build_name:
    continue

  info.model_header = model_headers.get(info.llm_build_name, "")

  if not info.model_header:
    print(f"warning: could not find model header for {arch}: {info.llm_build_name}")

  if False: # debug output
    print("\n\nMODEL_HEADER for {}:\n".format(arch))
    print(info.model_header)

    print("\n\nCODE_HPARAMS for {}:\n".format(arch))
    print(info.code_hparams)

    print("\n\nCODE_TENSORS for {}:\n".format(arch))
    print(info.code_tensors)





# remove info with empty llm_build_name
mapping = {arch: info for arch, info in mapping.items() if info.llm_build_name}






def add_indent(code: str, indent: str) -> str:
  return "\n".join(indent + line if line.strip() else line for line in code.splitlines())

def remove_indent(code: str, num_spaces: int) -> str:
  return "\n".join(line[num_spaces:] if len(line) > num_spaces else line for line in code.splitlines())

tmp = ""

for line in models_h_content.splitlines():
  tmp += line + "\n"
  if line == "// models":
    tmp += "//\n\n"
    break

tmp = tmp.replace('#include "llama-graph.h"',
                  '#include "llama-graph.h"\n#include "llama-model-loader.h"')

for arch, info in mapping.items():
  new_graph_struct = info.model_header
  use_base = "_base" in new_graph_struct
  new_graph_struct = new_graph_struct.replace(info.llm_build_name, "graph")
  if use_base:
    new_graph_struct = new_graph_struct.replace("public graph_base", "public " + info.llm_build_name + "_base")

  new_struct_code = """struct MODEL_NAME : public llm_arch_model_i {
    MODEL_NAME(const struct llama_model_params & params) : llm_arch_model_i(params) {};
    void load_hparams(llama_model_loader & ml) override;
    void load_tensors(llama_model_loader & ml) override;

GRAPH_STRUCT

    std::unique_ptr<llm_graph_context> build_graph_context(const llm_graph_params & params) const override;
};"""
  new_struct_code = new_struct_code.replace("MODEL_NAME", info.new_struct_name)
  new_struct_code = new_struct_code.replace("GRAPH_STRUCT", add_indent(new_graph_struct, "    "))

  tmp += "{}\n\n\n".format(new_struct_code)







# this will go to model file, but for debugging, we write to tmp for now
tmp += "\n\n\n\n\n\n"

for arch, info in mapping.items():
  new_model_code = """
void MODEL_NAME::load_hparams(llama_model_loader & ml) HPARAMS_CODE

void MODEL_NAME::load_tensors(llama_model_loader & ml) TENSORS_CODE

std::unique_ptr<llm_graph_context> MODEL_NAME::build_graph_context(const llm_graph_params & params) const GRAPH_CODE
"""
  code_hparams = info.code_hparams.strip()
  # if last line has break; we remove it
  if code_hparams.endswith("break;"):
    code_hparams = code_hparams[:-len("break;")].strip()
  
  code_hparams = remove_indent(code_hparams, 4*3)

  code_tensors = info.code_tensors.strip()
  # if last line has break; we remove it
  if code_tensors.endswith("break;"):
    code_tensors = code_tensors[:-len("break;")].strip()

  code_tensors = remove_indent(code_tensors, 4*4)
  # prepend "this->ml = ml;" to code_tensors
  code_tensors = "{\n    this->ml = &ml; // used by create_tensor\n\n" + "\n".join(code_tensors.splitlines()[1:])

  code_graph = info.code_graph.strip()
  # if last line has break; we remove it
  if code_graph.endswith("break;"):
    code_graph = code_graph[:-len("break;")].strip()
  code_graph = code_graph.replace("llm = ", "return ")
  code_graph = code_graph.replace(info.llm_build_name, "graph")

  new_model_code = new_model_code.replace("MODEL_NAME", info.new_struct_name)
  new_model_code = new_model_code.replace("HPARAMS_CODE", code_hparams)
  new_model_code = new_model_code.replace("TENSORS_CODE", code_tensors)
  new_model_code = new_model_code.replace("GRAPH_CODE", remove_indent(code_graph, 4*3))

  tmp += "{}\n\n\n".format(new_model_code)


with open("src/models/models_new.h", "w") as f:
  f.write(tmp)

