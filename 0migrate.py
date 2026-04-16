from __future__ import annotations

from dataclasses import dataclass, field
import re

# IMPORTANT: REMOVE THIS FILE BEFORE MERGING THE PR

# this script is partially vibe-coded

MODEL_H = "src/llama-model.h"
MODEL_CPP = "src/llama-model.cpp"

MARKER_START = "MARKER_START_MIGRATION_BUILD_GRAPH"
MARKER_END = "MARKER_END_MIGRATION_BUILD_GRAPH"

ARCH_RE = re.compile(r"case\s+(LLM_ARCH_[A-Z0-9_]+)\s*:")
BUILD_RE = re.compile(r"llm_build_[a-z0-9_]+")


@dataclass
class ModelInfo:
  LLM_ARCH: str
  llm_builds: set[str] = field(default_factory=set)
  handler_code: str = ""


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


def parse_build_graph_mapping(content: str) -> dict[str, ModelInfo]:
  region = extract_marked_region(content, MARKER_START, MARKER_END)
  lines = region.splitlines()

  mapping: dict[str, ModelInfo] = {}
  current_arches: list[str] = []
  current_handler_lines: list[str] = []

  switch_depth = 0
  inside_switch = False

  def finalize_current_handler() -> None:
    nonlocal current_arches, current_handler_lines
    if not current_arches:
      return

    handler_code = "\n".join(current_handler_lines).rstrip()
    llm_builds = set(BUILD_RE.findall(handler_code))

    for arch in current_arches:
      mapping[arch] = ModelInfo(
        LLM_ARCH=arch,
        llm_builds=llm_builds.copy(),
        handler_code=handler_code,
      )

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


mapping = parse_build_graph_mapping(model_cpp_content)

for arch, info in mapping.items():
  print(f"{arch} -> {sorted(info.llm_builds)}")
