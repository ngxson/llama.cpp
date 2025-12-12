import argparse
import requests
import json
from pathlib import Path
import logging

logger = logging.getLogger("compare-logits")
logging.basicConfig(level=logging.INFO)


DESCRIPTION = """
Compare logits between llama.cpp and another inference engine.

Example usage:
    Step 1: Dump logits from two different endpoints
        python scripts/compare-logits.py dump logits_llama.log http://localhost:8080/v1/completions
        python scripts/compare-logits.py dump logits_other.log http://other-engine:8000/v1/completions

        (optionally, you can add --api-key <key> if the endpoint requires authentication)

    Step 2: Compare the dumped logits
        python scripts/compare-logits.py compare logits_llama.log logits_other.log report.md
"""


INPUT_PROMPT = """
You are an advanced AI assistant capable of using tools to gather information, perform calculations, or execute tasks. Always think step by step before responding. If a user's query requires external data, computation, or actions beyond your internal knowledge, use the appropriate tools via function calls.

### Tool Call Format:
When you need to use a tool, output the call in this exact XML-inspired format. Include the opening and closing tags. Do not escape arguments; they will be parsed as plain text.

You can make multiple calls in one go by placing them one after another.
""".strip()


def dump_logits(endpoint: str, output_path: Path, api_key = None):
    logger.info(f"Dumping logits to {output_path} from endpoint {endpoint}...")
    words = INPUT_PROMPT.split(' ')
    curr_text = ""
    with output_path.open("w") as f:
        for i, w in enumerate(words):
            curr_text += (w + " ")
            payload = {
                "prompt": curr_text.strip(),
                "temperature": 0.0,
                "top_k": 1,
                "max_tokens": 1,
                "logprobs": 1,
                "stream": False,
            }
            response = requests.post(endpoint,
                                     json=payload,
                                     headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
            response.raise_for_status()
            data = response.text
            f.write(f"{data}\n")
            logger.info(f"\n\n{data}\n\n[{i + 1}/{len(words)}]")
    logger.info(f"Logits dumped to {output_path}")


def get_token_logprobs(data: dict):
    logprobs = data['choices'][0]['logprobs']
    if 'content' in logprobs:
        # llama.cpp case
        top = logprobs['content'][0]['top_logprobs'][0]
        return top['token'], top['logprob']
    else:
        # vllm case
        tokens = logprobs['tokens']
        token_logprobs = logprobs['token_logprobs']
        return tokens[0], token_logprobs[0]


def clean_text(text: str) -> str:
    return "'" + text.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r").replace("|", "\\|") + "'"


def compare_logits(input1: Path, input2: Path, output_path: Path):
    with input1.open("r") as f1, input2.open("r") as f2, output_path.open("w") as fout:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

        tab_header = [input1.name, "logprob_1",  input2.name, "logprob_2", "diff (abs)"]
        tab_entries = []
        tab_max_widths = [len(h) for h in tab_header]

        assert len(lines1) == len(lines2), "Input files must have the same number of lines."

        fout.write("# Logits Comparison Report\n\n")
        for i, (line1, line2) in enumerate(zip(lines1, lines2)):
            if not line1.strip() or not line2.strip():
                continue # skip empty lines

            data1 = json.loads(line1)
            data2 = json.loads(line2)

            token1, logprob1 = get_token_logprobs(data1)
            token2, logprob2 = get_token_logprobs(data2)

            token1 = clean_text(token1)
            token2 = clean_text(token2)
            abs_diff = abs(logprob1 - logprob2)

            tab_entries.append((token1, f"{logprob1:.4f}", token2, f"{logprob2:.4f}", f"{(abs_diff):.4f}"))

        for i in range(len(tab_entries)):
            for j in range(len(tab_header)):
                tab_max_widths[j] = max(tab_max_widths[j], len(tab_entries[i][j]))

        output = ""
        for j in range(len(tab_header)):
            output += f"| {tab_header[j]:<{tab_max_widths[j]}} "
        output += "|\n"
        for j in range(len(tab_header)):
            output += f"|{'-' * (tab_max_widths[j] + 2)}"
        output += "|\n"
        for entry in tab_entries:
            for j in range(len(tab_header)):
                output += f"| {entry[j]:<{tab_max_widths[j]}} "
            output += "|\n"

        logger.info("\n" + output)
        fout.write(output)
        logger.info(f"Report written to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="verb", required=True, help="action to perform")

    # dump subcommand
    parser_dump = subparsers.add_parser("dump", help="dump logits from an endpoint")
    parser_dump.add_argument("output", type=Path, help="output path for dumped logits (.log)")
    parser_dump.add_argument("endpoint", type=str, help="OAI-compat /completions endpoint")
    parser_dump.add_argument("--api-key", type=str, default=None, help="API key for authentication (if required)")
    parser_dump.add_argument("--file", type=Path, default=None, help="File containing prompt to use instead of the default")

    # compare subcommand
    parser_compare = subparsers.add_parser("compare", help="compare two dumped logits files")
    parser_compare.add_argument("input1", type=Path, help="first input file (.log)")
    parser_compare.add_argument("input2", type=Path, help="second input file (.log)")
    parser_compare.add_argument("output", type=Path, help="output path for comparison report (.md)")

    try:
        return parser.parse_args()
    except Exception as e:
        parser.print_help()
        raise e


def main():
    args = parse_args()

    if args.verb == "dump":
        if args.file is not None:
            with args.file.open("r") as f:
                global INPUT_PROMPT
                INPUT_PROMPT = f.read().strip()
        dump_logits(args.endpoint, args.output, args.api_key)
    elif args.verb == "compare":
        compare_logits(args.input1, args.input2, args.output)
    else:
        raise ValueError(f"Unknown verb: {args.verb}")


if __name__ == "__main__":
    main()
