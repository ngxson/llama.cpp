import sse from 'k6/x/sse'
import {check, sleep} from 'k6'
import {SharedArray} from 'k6/data'
import {Counter, Rate, Trend} from 'k6/metrics'
import exec from 'k6/execution';

// Number of virtual users
const n_uvs = 16;

// Server chat completions prefix
const server_url = __ENV.SERVER_BENCH_URL ? __ENV.SERVER_BENCH_URL : 'http://localhost:8080/v1'

// Number of total prompts in the dataset - default 10m / 10 seconds/request * number of users
const n_prompt = __ENV.SERVER_BENCH_N_PROMPTS ? parseInt(__ENV.SERVER_BENCH_N_PROMPTS) : 600 / 10 * NUMBER_UVS

// Model name to request
const model = __ENV.SERVER_BENCH_MODEL_ALIAS ? __ENV.SERVER_BENCH_MODEL_ALIAS : 'my-model'

// Dataset path
const dataset_path = __ENV.SERVER_BENCH_DATASET ? __ENV.SERVER_BENCH_DATASET : './ShareGPT_V3_unfiltered_cleaned_split.json'

// Max tokens to predict
const max_tokens = __ENV.SERVER_BENCH_MAX_TOKENS ? parseInt(__ENV.SERVER_BENCH_MAX_TOKENS) : 512

// Max prompt tokens
const n_prompt_tokens = __ENV.SERVER_BENCH_MAX_PROMPT_TOKENS ? parseInt(__ENV.SERVER_BENCH_MAX_PROMPT_TOKENS) : 1024

// Max slot context
const n_ctx_slot = __ENV.SERVER_BENCH_MAX_CONTEXT ? parseInt(__ENV.SERVER_BENCH_MAX_CONTEXT) : 2048

export function setup() {
    console.info(`Benchmark config: server_url=${server_url} n_prompt=${n_prompt} model=${model} dataset_path=${dataset_path} max_tokens=${max_tokens}`)
}

const data = new SharedArray('conversations', function () {
    const tokenizer = (message) => message.split(/[\s,'".?]/)

    return JSON.parse(open(dataset_path))
        // Filter out the conversations with less than 2 turns.
        .filter(data => data["conversations"].length >= 2)
        .filter(data => data["conversations"][0]["from"] === "human")
        .map(data => {
            return {
                prompt: data["conversations"][0]["value"],
                n_prompt_tokens: tokenizer(data["conversations"][0]["value"]).length,
                n_completion_tokens: tokenizer(data["conversations"][1]["value"]).length,
            }
        })
        // Filter out too short sequences
        .filter(conv => conv.n_prompt_tokens >= 4 && conv.n_completion_tokens >= 4)
        // Filter out too long sequences.
        .filter(conv => conv.n_prompt_tokens <= n_prompt_tokens && conv.n_prompt_tokens + conv.n_completion_tokens <= n_ctx_slot)
        // Keep only first n prompts
        .slice(0, n_prompt)
})

const metric_prompt_tokens = new Trend('metric_prompt_tokens')
const metric_completion_tokens = new Trend('metric_completion_tokens')

const metric_tokens_second = new Trend('metric_tokens_second')
const metric_prompt_processing_second = new Trend('metric_prompt_processing_second')

const metric_prompt_tokens_total_counter = new Counter('metric_prompt_tokens_total_counter')
const metric_completion_tokens_total_counter = new Counter('metric_completion_tokens_total_counter')

const metric_completions_truncated_rate = new Rate('metric_completions_truncated_rate')
const metric_completions_stop_rate = new Rate('metric_completions_stop_rate')

export const options = {
    thresholds: {
        metric_completions_truncated_rate: [
            // more than 80% of truncated input will abort the test
            //{threshold: 'rate < 0.8', abortOnFail: true, delayAbortEval: '1m'},
        ],
    },
    executor: 'constant-vus',
    duration: '10m',
    vus: n_uvs,
}

export default function () {
    const conversation = data[exec.scenario.iterationInInstance % data.length]
    const payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are ChatGPT, an AI assistant.",
            },
            {
                "role": "user",
                "content": conversation.prompt,
            }
        ],
        "model": model,
        "stream": true,
        //"seed": 42,
        "max_tokens": max_tokens,
        //"stop": ["<|im_end|>"] // This is temporary for phi-2 base (i.e. not instructed) since the server expects that the model always to emit BOS
    }

    const params = {method: 'POST', body: JSON.stringify(payload), headers: {
        'Content-Type': 'application/json'
    }};

    const startTime = new Date()
    let promptEvalEndTime = null
    let prompt_tokens = 0
    let completions_tokens = 0
    let finish_reason = null
    const res = sse.open(`${server_url}/chat/completions`, params, function (client) {
        client.on('event', function (event) {
            if (promptEvalEndTime == null) {
                promptEvalEndTime = new Date()
            }

            if (event.data == '[DONE]') {
                return
            }
            let chunk = JSON.parse(event.data)
            let choice = chunk.choices[0]
            if (choice.finish_reason) {
                finish_reason = choice.finish_reason
            }

            if (chunk.usage) {
                prompt_tokens = chunk.usage.prompt_tokens
                metric_prompt_tokens.add(prompt_tokens)
                metric_prompt_tokens_total_counter.add(prompt_tokens)

                completions_tokens = chunk.usage.completion_tokens
                metric_completion_tokens.add(completions_tokens)
                metric_completion_tokens_total_counter.add(completions_tokens)
            }
        })

        client.on('error', function (e) {
            console.log('An unexpected error occurred: ', e.error());
            throw e;
        })
    })

    check(res, {'success completion': (r) => r.status === 200})

    const endTime = new Date()

    const promptEvalTime = promptEvalEndTime - startTime
    if (promptEvalTime > 0) {
        metric_prompt_processing_second.add(prompt_tokens / (promptEvalEndTime - startTime) * 1.e3)
    }

    const completion_time = endTime - promptEvalEndTime
    if (completions_tokens > 0 && completion_time > 0) {
        metric_tokens_second.add(completions_tokens / completion_time * 1.e3)
    }
    metric_completions_truncated_rate.add(finish_reason === 'length')
    metric_completions_stop_rate.add(finish_reason === 'stop')

    sleep(0.3)
}
