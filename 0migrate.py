from __future__ import annotations

from dataclasses import dataclass, field
import re
import os

# IMPORTANT: REMOVE THIS FILE BEFORE MERGING THE PR

# this script is partially vibe-coded

MODEL_H = "src/llama-model.h"
MODEL_CPP = "src/llama-model.cpp"

# Reset any changes in src/models before running to avoid reading already modified files
print("Resetting src/models/* ...")
os.system("git checkout src/models")
os.system("git checkout " + MODEL_CPP)
os.system("git clean -fd src/models")

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
  code_impl: str = ""
  model_header: str = ""
  reuse_graph_from_arch: str = ""
  reuse_graph_from_model: str = ""
  reuse_hparams_from_arch: str = ""
  reuse_hparams_from_model: str = ""
  reuse_tensors_from_arch: str = ""
  reuse_tensors_from_model: str = ""
  # transformed code
  new_struct_name: str = ""
  new_header: str = ""
  new_impl: str = ""


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
    raise ValueError(f"expected exactly one llm_build for {arch}, found: {info.llm_builds}")

  info.llm_build_name = next(iter(info.llm_builds))
  print(f"{arch} -> {info.llm_build_name}")
  info.new_struct_name = info.llm_build_name.replace("llm_build_", "llama_model_")

graph_owner_by_build_name: dict[str, str] = {}
hparams_owner_by_code: dict[str, str] = {}
tensors_owner_by_code: dict[str, str] = {}

for arch, info in mapping.items():
  if not info.llm_build_name:
    continue

  # graph reuse
  owner_arch = graph_owner_by_build_name.get(info.llm_build_name)
  if owner_arch is None:
    graph_owner_by_build_name[info.llm_build_name] = arch
  else:
    info.reuse_graph_from_arch = owner_arch
    info.reuse_graph_from_model = mapping[owner_arch].new_struct_name

  # hparams reuse
  hcode = info.code_hparams.strip()
  if hcode:
    hparams_owner = hparams_owner_by_code.get(hcode)
    if hparams_owner is None:
      hparams_owner_by_code[hcode] = arch
    elif hparams_owner != arch:
      info.reuse_hparams_from_arch = hparams_owner
      info.reuse_hparams_from_model = mapping[hparams_owner].new_struct_name

  # tensors reuse
  tcode = info.code_tensors.strip()
  if tcode:
    tensors_owner = tensors_owner_by_code.get(tcode)
    if tensors_owner is None:
      tensors_owner_by_code[tcode] = arch
    elif tensors_owner != arch:
      info.reuse_tensors_from_arch = tensors_owner
      info.reuse_tensors_from_model = mapping[tensors_owner].new_struct_name





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

# print("\n\nSELECT_ARCH_FN:\n")
# print(output_select_arch_fn)

model_cpp_content = model_cpp_content.replace("// SELECT_ARCH_FN", output_select_arch_fn)






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



# one-off hotfix
for arch, info in mapping.items():
  if arch == "LLM_ARCH_T5ENCODER":
    info.reuse_graph_from_arch = "LLM_ARCH_T5"
    info.reuse_graph_from_model = mapping["LLM_ARCH_T5"].new_struct_name
    print(f"hotfix: {arch} will reuse graph from {info.reuse_graph_from_arch} ({info.reuse_graph_from_model})")






def add_indent(code: str, indent: str) -> str:
  return "\n".join(indent + line if line.strip() else line for line in code.splitlines())

def remove_indent(code: str, num_spaces: int) -> str:
  return "\n".join(line[num_spaces:] if len(line) > num_spaces else line for line in code.splitlines())



seen = set()
for arch, info in mapping.items():
  if info.new_struct_name in seen:
    nnn = arch.replace("LLM_ARCH_", "").lower()
    new_name = "llama_model_" + nnn
    print(f"warning: duplicate {info.new_struct_name}, renamed to {new_name}")
    info.new_struct_name = new_name
  seen.add(info.new_struct_name)

  fname = info.new_struct_name.replace("llama_model_", "").replace("_", "-")
  impl_filename = f"src/models/{fname}.cpp"
  if os.path.exists(impl_filename):
    with open(impl_filename, "r") as f_impl:
      info.code_impl = f_impl.read()

  if info.reuse_graph_from_model:
    if info.model_header and info.model_header.strip().startswith("template"):
      template_line = info.model_header.strip().splitlines()[0]
      
      template_args_match = re.search(r'<(.+)>', template_line)
      if template_args_match:
        args_str = template_args_match.group(1)
        args_list = []
        for arg in args_str.split(','):
          name = arg.strip().split()[-1]
          args_list.append(name)
        args_joined = ", ".join(args_list)
        new_graph_struct = template_line + "\nusing graph = " + info.reuse_graph_from_model + "::graph<" + args_joined + ">;"
      else:
        new_graph_struct = template_line + "\nusing graph = " + info.reuse_graph_from_model + "::graph;"
    else:
      new_graph_struct = "using graph = " + info.reuse_graph_from_model + "::graph;"
  else:
    new_graph_struct = info.model_header
    use_base = "_base" in new_graph_struct
    new_graph_struct = new_graph_struct.replace(info.llm_build_name, "graph")
    if use_base:
      new_graph_struct = new_graph_struct.replace("public graph_base", "public " + info.llm_build_name + "_base")

  base_class = "llm_arch_model_i"
  load_methods_decl = """    void load_hparams(llama_model_loader & ml) override;\n    void load_tensors(llama_model_loader & ml) override;"""
  if info.reuse_hparams_from_model and info.reuse_hparams_from_model == info.reuse_tensors_from_model:
    base_class = info.reuse_hparams_from_model
    load_methods_decl = "    // reuse load_hparams and load_tensors from {}".format(info.reuse_hparams_from_model)

  new_struct_code = """struct MODEL_NAME : public BASE_CLASS {
    MODEL_NAME(const struct llama_model_params & params) : BASE_CLASS(params) {}
LOAD_METHODS_DECL

GRAPH_STRUCT

    std::unique_ptr<llm_graph_context> build_graph_context(const llm_graph_params & params) const override;
};"""
  new_struct_code = new_struct_code.replace("MODEL_NAME", info.new_struct_name)
  new_struct_code = new_struct_code.replace("BASE_CLASS", base_class)
  new_struct_code = new_struct_code.replace("LOAD_METHODS_DECL\n", load_methods_decl + "\n" if load_methods_decl else "")
  new_struct_code = new_struct_code.replace("GRAPH_STRUCT", add_indent(new_graph_struct, "    "))
  info.new_header = new_struct_code









for arch, info in mapping.items():
  new_model_code = """
void MODEL_NAME::load_hparams(llama_model_loader & ml) HPARAMS_CODE

void MODEL_NAME::load_tensors(llama_model_loader & ml) TENSORS_CODE

std::unique_ptr<llm_graph_context> MODEL_NAME::build_graph_context(const llm_graph_params & params) const GRAPH_CODE
"""
  if info.reuse_hparams_from_model and info.reuse_hparams_from_model == info.reuse_tensors_from_model:
    print(f"{arch} will reuse hparams and tensors from {info.reuse_hparams_from_arch} ({info.reuse_hparams_from_model})")
    new_model_code = """
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

  if info.reuse_graph_from_model:
    print(f"{arch} will reuse graph from {info.reuse_graph_from_arch} ({info.reuse_graph_from_model})")

  code_graph = code_graph.replace("llm = ", "return ")
  #code_graph = code_graph.replace(info.llm_build_name, info.new_struct_name + "::graph")
  code_graph = code_graph.replace(info.llm_build_name, "graph")

  new_model_code = new_model_code.replace("MODEL_NAME", info.new_struct_name)
  if "HPARAMS_CODE" in new_model_code:
    new_model_code = new_model_code.replace("HPARAMS_CODE", code_hparams)
  if "TENSORS_CODE" in new_model_code:
    new_model_code = new_model_code.replace("TENSORS_CODE", code_tensors)
  new_model_code = new_model_code.replace("GRAPH_CODE", remove_indent(code_graph, 4*3))
  info.new_impl = new_model_code





# assemble the new impl code
for arch, info in mapping.items():
  new_impl = info.new_impl.strip()
  code_impl = info.code_impl.strip().splitlines()
  # split code and include sections from code_impl
  code_includes = []
  code_impl_lines = []
  for line in code_impl:
    if line.strip().startswith("#include"):
      code_includes.append(line)
    else:
      code_impl_lines.append(line)
  code_includes = "\n".join(code_includes).strip()
  code_impl = "\n".join(code_impl_lines).strip()
  # if no code_includes, make one
  code_includes = '#include "models.h"' if not code_includes else code_includes
  info.new_impl = code_includes + "\n" + info.new_impl + "\n" + code_impl
  # normalize
  info.new_impl = info.new_impl.replace(" ::", "::")
  # rename graph building in impl
  # handles template: llm_build_plamo3<iswa>::llm_build_plamo3 -> llama_model_plamo3::graph<iswa>::graph
  info.new_impl = re.sub(
    info.llm_build_name + r"(<[^>]+>)?::" + info.llm_build_name,
    info.new_struct_name + r"::graph\1::graph",
    info.new_impl
  )
  # handles: str llm_build_plamo3<true> -> str llama_model_plamo3::graph<true>
  info.new_impl = re.sub(
    r'\b' + info.llm_build_name + r'(<[^>]*>)',
    info.new_struct_name + r"::graph\1",
    info.new_impl
  )
  # handles: llm_build_plamo3:: -> llama_model_plamo3::graph::
  info.new_impl = re.sub(
    r'\b' + info.llm_build_name + r'::',
    info.new_struct_name + r"::graph::",
    info.new_impl
  )
  # make sure to add a trailing newline
  if not info.new_impl.endswith("\n"):
    info.new_impl += "\n"


  if arch == "LLM_ARCH_T5ENCODER":
    info.new_header = info.new_header.replace("llama_model_t5::graph", "llama_model_t5::graph<true>")
    new_impl = info.new_impl.splitlines()
    new_impl = [line for line in new_impl if "::graph" not in line]
    info.new_impl = "\n".join(new_impl).strip() + "\n"





header_file = ""
for line in models_h_content.splitlines():
  header_file += line + "\n"
  if line == "// models":
    header_file += "//"
    break
header_file = header_file.replace('#include "llama-graph.h"',
                                  '#include "llama-graph.h"\n#include "llama-model-loader.h"')
for arch, info in mapping.items():
  header_file += "\n\n" + info.new_header + "\n"

tmp_impl = ""
tmp_impl += "\n\n\n\n\n\n"
for arch, info in mapping.items():
  tmp_impl += info.new_impl + "\n"



DO_IT_FOR_REAL = True
if DO_IT_FOR_REAL:
  # remove all from src/models/*.cpp except base classes
  os.system("find src/models -name '*.cpp' ! -name '*-base.cpp' -type f -delete")
  with open("src/models/models.h", "w") as f:
    f.write(header_file)
  for arch, info in mapping.items():
    fname = info.new_struct_name
    fname = fname.replace("llama_model_", "")
    fname = fname.replace("_", "-")
    impl_filename = f"src/models/{fname}.cpp"
    with open(impl_filename, "w") as f:
      f.write(info.new_impl)
      # print("writing {}...".format(impl_filename))
    with open(MODEL_CPP, "w") as f:
      f.write(model_cpp_content)
else:
  with open("src/models/models_new.h", "w") as f:
    f.write(header_file + tmp_impl)

