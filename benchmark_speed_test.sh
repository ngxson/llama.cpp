#!/bin/bash
# Qwen3-0.6B Quantization Speed Benchmark Script
# Runs llama-bench multiple times per model and calculates statistics

# Note: Not using 'set -e' as we handle errors explicitly

# Default configuration
ITERATIONS=100
THREADS=4
REPEATS=3
PROMPT_TOKENS=0
GENERATE_TOKENS=20

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        -t|--threads)
            THREADS="$2"
            shift 2
            ;;
        -r|--repeats)
            REPEATS="$2"
            shift 2
            ;;
        -p|--prompt-tokens)
            PROMPT_TOKENS="$2"
            shift 2
            ;;
        -n|--generate-tokens)
            GENERATE_TOKENS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -i, --iterations N      Number of iterations per model (default: 100)"
            echo "  -t, --threads N         Number of threads (default: 4)"
            echo "  -r, --repeats N         Repeats per run (default: 3)"
            echo "  -p, --prompt-tokens N   Prompt tokens (default: 0)"
            echo "  -n, --generate-tokens N Generate tokens (default: 20)"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Configuration
LLAMA_BENCH="./build/bin/llama-bench"
declare -a MODEL_NAMES=("Q3_K_S" "Q3_K_M" "Q3_HIFI")
declare -a MODEL_PATHS=(
    "./Qwen3-0.6B-f16:Q3_K_S.gguf"
    "./Qwen3-0.6B-f16:Q3_K_M.gguf"
    "./Qwen3-0.6B-f16:Q3_HIFI.gguf"
)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;90m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Verify files exist
if [[ ! -x "$LLAMA_BENCH" ]]; then
    echo -e "${RED}Error: llama-bench not found or not executable at: $LLAMA_BENCH${NC}"
    exit 1
fi

for i in "${!MODEL_PATHS[@]}"; do
    if [[ ! -f "${MODEL_PATHS[$i]}" ]]; then
        echo -e "${RED}Error: Model not found: ${MODEL_PATHS[$i]}${NC}"
        exit 1
    fi
done

# Results storage - using temp files for arrays
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

for name in "${MODEL_NAMES[@]}"; do
    touch "$TEMP_DIR/${name}_speeds.txt"
    echo "0" > "$TEMP_DIR/${name}_errors.txt"
done

# Print header
print_line() {
    printf '=%.0s' {1..70}
    echo ""
}

print_dash() {
    printf -- '-%.0s' {1..70}
    echo ""
}

echo -e "${CYAN}"
print_line
echo "QWEN3-14B QUANTIZATION SPEED BENCHMARK"
print_line
echo -e "${NC}"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Iterations per model: $ITERATIONS"
echo "  Threads: $THREADS"
echo "  Repeats per run: $REPEATS"
echo "  Generate tokens: $GENERATE_TOKENS"
echo "  Models: ${#MODEL_NAMES[@]}"
echo ""

START_TIME=$(date +%s)
TOTAL_RUNS=$((ITERATIONS * ${#MODEL_NAMES[@]}))

echo -e "${GREEN}Starting benchmark at $(date '+%H:%M:%S')...${NC}"
EST_MINUTES=$(echo "scale=1; $TOTAL_RUNS * 5 / 60" | bc)
echo -e "${GRAY}Total runs: $TOTAL_RUNS (estimated time: ${EST_MINUTES} minutes)${NC}"
echo ""

# Progress tracking
CURRENT_RUN=0

# Function to display progress bar
show_progress() {
    local current=$1
    local total=$2
    local model=$3
    local iteration=$4
    local percent=$((current * 100 / total))
    local filled=$((percent / 2))
    local empty=$((50 - filled))
    
    # Build progress bar string (handle edge cases where filled or empty is 0)
    local bar=""
    if [[ $filled -gt 0 ]]; then
        bar=$(printf '#%.0s' $(seq 1 $filled))
    fi
    if [[ $empty -gt 0 ]]; then
        bar="${bar}$(printf ' %.0s' $(seq 1 $empty))"
    fi
    
    printf "\r[%-50s] %3d%% - %s iter %d/%d" "$bar" "$percent" "$model" "$iteration" "$ITERATIONS"
}

# Main benchmark loop
for ((i = 1; i <= ITERATIONS; i++)); do
    for idx in "${!MODEL_NAMES[@]}"; do
        name="${MODEL_NAMES[$idx]}"
        path="${MODEL_PATHS[$idx]}"
        
        CURRENT_RUN=$((CURRENT_RUN + 1))
        
        # Show progress
        show_progress $CURRENT_RUN $TOTAL_RUNS "$name" $i
        
        # Run benchmark and capture output
        output=$("$LLAMA_BENCH" -m "$path" -t "$THREADS" -r "$REPEATS" -p "$PROMPT_TOKENS" -n "$GENERATE_TOKENS" 2>&1) || true
        
        # Parse output - look for tg (token generation) speed
        # Format: | model | size | params | backend | threads | test | t/s |
        # Example: | qwen3 4B Q3_K - Small | 948.91 MiB | 2.03 B | CPU | 4 | tg20 | 28.87 ± 1.45 |
        found=false
        
        while IFS= read -r line; do
            # Match pattern: anything with tg followed by speed ± stddev
            if [[ $line =~ tg[0-9]+[[:space:]]*\|[[:space:]]*([0-9.]+)[[:space:]]*± ]]; then
                speed="${BASH_REMATCH[1]}"
                echo "$speed" >> "$TEMP_DIR/${name}_speeds.txt"
                found=true
                break
            # Alternative pattern: just numbers at end
            elif [[ $line =~ \|[[:space:]]*tg[0-9]+[[:space:]]*\|[[:space:]]*([0-9.]+) ]]; then
                speed="${BASH_REMATCH[1]}"
                echo "$speed" >> "$TEMP_DIR/${name}_speeds.txt"
                found=true
                break
            fi
        done <<< "$output"
        
        if [[ $found == false ]]; then
            # Debug: show what we got if parsing failed on first iteration
            if [[ $i -eq 1 ]]; then
                echo ""
                echo -e "${GRAY}  Debug - Raw output sample for $name:${NC}"
                echo "$output" | head -10 | while read -r line; do
                    echo -e "${GRAY}    $line${NC}"
                done
            fi
            errors=$(cat "$TEMP_DIR/${name}_errors.txt")
            echo $((errors + 1)) > "$TEMP_DIR/${name}_errors.txt"
        fi
    done
    
    # Periodic status update every 10 iterations
    if ((i % 10 == 0)); then
        NOW=$(date +%s)
        ELAPSED=$((NOW - START_TIME))
        ELAPSED_FMT=$(printf '%02d:%02d:%02d' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))
        
        if [[ $CURRENT_RUN -gt 0 ]]; then
            REMAINING=$(( (ELAPSED * (TOTAL_RUNS - CURRENT_RUN)) / CURRENT_RUN ))
            REMAINING_FMT=$(printf '%02d:%02d:%02d' $((REMAINING/3600)) $((REMAINING%3600/60)) $((REMAINING%60)))
        else
            REMAINING_FMT="--:--:--"
        fi
        
        echo ""
        echo -e "${GRAY}  [$i/$ITERATIONS] Elapsed: $ELAPSED_FMT | ETA: $REMAINING_FMT${NC}"
    fi
done

echo ""
echo ""

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_FMT=$(printf '%02d:%02d:%02d' $((DURATION/3600)) $((DURATION%3600/60)) $((DURATION%60)))

# Calculate statistics function
calc_stats() {
    local name=$1
    local file="$TEMP_DIR/${name}_speeds.txt"
    
    if [[ ! -s "$file" ]]; then
        echo "0 0 0 0 0 0 0 0"
        return
    fi
    
    # Sort the data
    sort -n "$file" > "$TEMP_DIR/${name}_sorted.txt"
    local count=$(wc -l < "$TEMP_DIR/${name}_sorted.txt")
    
    if [[ $count -eq 0 ]]; then
        echo "0 0 0 0 0 0 0 0"
        return
    fi
    
    # Calculate statistics using awk
    awk -v count="$count" '
    BEGIN { sum = 0; sumsq = 0 }
    {
        values[NR] = $1
        sum += $1
        sumsq += $1 * $1
    }
    END {
        mean = sum / count
        variance = (sumsq / count) - (mean * mean)
        stddev = sqrt(variance > 0 ? variance : 0)
        
        # Min and Max
        min = values[1]
        max = values[count]
        
        # Median
        mid = int(count / 2)
        if (count % 2 == 0) {
            median = (values[mid] + values[mid + 1]) / 2
        } else {
            median = values[mid + 1]
        }
        
        # Percentiles
        p5_idx = int(count * 0.05) + 1
        p95_idx = int(count * 0.95)
        if (p95_idx < 1) p95_idx = 1
        if (p95_idx > count) p95_idx = count
        
        p5 = values[p5_idx]
        p95 = values[p95_idx]
        
        printf "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %d\n", mean, stddev, median, min, max, p5, p95, count
    }' "$TEMP_DIR/${name}_sorted.txt"
}

# Generate report
echo -e "${CYAN}"
print_line
echo "BENCHMARK RESULTS"
print_line
echo -e "${NC}"

echo -e "${GREEN}Test completed in: $DURATION_FMT${NC}"
echo "Total iterations per model: $ITERATIONS"
echo ""

# Collect all stats
declare -A STATS
FASTEST_MEAN=0

for name in "${MODEL_NAMES[@]}"; do
    stats=$(calc_stats "$name")
    STATS[$name]="$stats"
    mean=$(echo "$stats" | awk '{print $1}')
    if (( $(echo "$mean > $FASTEST_MEAN" | bc -l) )); then
        FASTEST_MEAN=$mean
    fi
done

# Detailed results table
echo -e "${YELLOW}SPEED COMPARISON (tokens/second - higher is better)${NC}"
print_dash

printf "${WHITE}%-15s %10s %10s %10s %10s %10s %10s${NC}\n" "Model" "Mean" "StdDev" "Median" "Min" "Max" "vs Best"
print_dash

for name in "${MODEL_NAMES[@]}"; do
    read -r mean stddev median min max p5 p95 count <<< "${STATS[$name]}"
    
    if (( $(echo "$mean == $FASTEST_MEAN" | bc -l) )); then
        vs_best="FASTEST"
        color="${GREEN}"
    else
        diff_pct=$(echo "scale=1; (1 - $mean / $FASTEST_MEAN) * 100" | bc)
        vs_best="-${diff_pct}%"
        color="${NC}"
    fi
    
    printf "${color}%-15s %10.2f %10.2f %10.2f %10.2f %10.2f %10s${NC}\n" \
        "$name" "$mean" "$stddev" "$median" "$min" "$max" "$vs_best"
done

print_dash
echo ""

# Percentile analysis
echo -e "${YELLOW}PERCENTILE ANALYSIS${NC}"
print_dash
printf "${WHITE}%-15s %12s %12s %12s %10s${NC}\n" "Model" "5th %ile" "Median" "95th %ile" "Samples"
print_dash

for name in "${MODEL_NAMES[@]}"; do
    read -r mean stddev median min max p5 p95 count <<< "${STATS[$name]}"
    errors=$(cat "$TEMP_DIR/${name}_errors.txt")
    
    printf "%-15s %12.2f %12.2f %12.2f %10s\n" \
        "$name" "$p5" "$median" "$p95" "$count/$ITERATIONS"
done

print_dash
echo ""

# Speed ranking summary
echo -e "${YELLOW}SPEED RANKING SUMMARY${NC}"
print_dash

# Create ranking array
declare -a RANKING
for name in "${MODEL_NAMES[@]}"; do
    mean=$(echo "${STATS[$name]}" | awk '{print $1}')
    RANKING+=("$mean|$name")
done

# Sort by mean (descending)
IFS=$'\n' SORTED_RANKING=($(sort -t'|' -k1 -nr <<< "${RANKING[*]}"))
unset IFS

RANK=1
FIRST_MEAN=""

for entry in "${SORTED_RANKING[@]}"; do
    mean=$(echo "$entry" | cut -d'|' -f1)
    name=$(echo "$entry" | cut -d'|' -f2)
    stddev=$(echo "${STATS[$name]}" | awk '{print $2}')
    
    if [[ $RANK -eq 1 ]]; then
        FIRST_MEAN=$mean
        speed_diff=""
    else
        diff_tps=$(echo "scale=2; $FIRST_MEAN - $mean" | bc)
        diff_pct=$(echo "scale=1; ($diff_tps / $FIRST_MEAN) * 100" | bc)
        speed_diff="($diff_tps t/s slower, -${diff_pct}%)"
    fi
    
    case $RANK in
        1) medal="🥇" ;;
        2) medal="🥈" ;;
        3) medal="🥉" ;;
        *) medal="  " ;;
    esac
    
    mean_fmt=$(printf "%.2f" "$mean")
    stddev_fmt=$(printf "%.2f" "$stddev")
    
    echo "$medal #$RANK $name: $mean_fmt ± $stddev_fmt t/s $speed_diff"
    RANK=$((RANK + 1))
done

echo ""
print_line

# Export results to CSV
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
CSV_PATH="benchmark_results_${TIMESTAMP}.csv"

echo "Model,Mean_TPS,StdDev,Median,Min,Max,P5,P95,Samples,Errors" > "$CSV_PATH"
for name in "${MODEL_NAMES[@]}"; do
    read -r mean stddev median min max p5 p95 count <<< "${STATS[$name]}"
    errors=$(cat "$TEMP_DIR/${name}_errors.txt")
    echo "$name,$mean,$stddev,$median,$min,$max,$p5,$p95,$count,$errors" >> "$CSV_PATH"
done

echo -e "${GREEN}Results exported to: $CSV_PATH${NC}"

# Export raw data to JSON
RAW_PATH="benchmark_raw_${TIMESTAMP}.json"
echo "{" > "$RAW_PATH"
first=true
for name in "${MODEL_NAMES[@]}"; do
    if [[ $first == true ]]; then
        first=false
    else
        echo "," >> "$RAW_PATH"
    fi
    printf '  "%s": [' "$name" >> "$RAW_PATH"
    
    # Read speeds and format as JSON array
    if [[ -s "$TEMP_DIR/${name}_speeds.txt" ]]; then
        paste -sd, "$TEMP_DIR/${name}_speeds.txt" >> "$RAW_PATH"
    fi
    
    printf ']' >> "$RAW_PATH"
done
echo "" >> "$RAW_PATH"
echo "}" >> "$RAW_PATH"

echo -e "${GREEN}Raw data exported to: $RAW_PATH${NC}"

