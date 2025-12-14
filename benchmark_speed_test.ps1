# Qwen3-1.7B Quantization Speed Benchmark Script
# Runs llama-bench 100 times per model and calculates statistics

param(
    [int]$Iterations = 100,
    [int]$Threads = 4,
    [int]$Repeats = 3,
    [int]$PromptTokens = 0,
    [int]$GenerateTokens = 20
)

$ErrorActionPreference = "Stop"

# Configuration
$LlamaBench = ".\build\bin\Release\llama-bench.exe"
$Models = @(
    @{ Name = "Q3_K_S"; Path = ".\Qwen3-1.7B-f16-Q3_K_S.gguf" },
    @{ Name = "Q3_K_M"; Path = ".\Qwen3-1.7B-f16-Q3_K_M.gguf" },
    @{ Name = "Q3_HIFI"; Path = ".\Qwen3-1.7B-f16-Q3_HIFI.gguf" }
)

# Verify files exist
if (-not (Test-Path $LlamaBench)) {
    Write-Error "llama-bench not found at: $LlamaBench"
    exit 1
}

foreach ($model in $Models) {
    if (-not (Test-Path $model.Path)) {
        Write-Error "Model not found: $($model.Path)"
        exit 1
    }
}

# Results storage
$Results = @{}
foreach ($model in $Models) {
    $Results[$model.Name] = @{
        Speeds = [System.Collections.ArrayList]::new()
        Errors = 0
    }
}

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "QWEN3-1.7B QUANTIZATION SPEED BENCHMARK" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Iterations per model: $Iterations"
Write-Host "  Threads: $Threads"
Write-Host "  Repeats per run: $Repeats"
Write-Host "  Generate tokens: $GenerateTokens"
Write-Host "  Models: $($Models.Count)"
Write-Host ""

$StartTime = Get-Date
$TotalRuns = $Iterations * $Models.Count

Write-Host "Starting benchmark at $($StartTime.ToString('HH:mm:ss'))..." -ForegroundColor Green
Write-Host "Total runs: $TotalRuns (estimated time: $([math]::Round($TotalRuns * 5 / 60, 1)) minutes)" -ForegroundColor Gray
Write-Host ""

# Progress tracking
$CurrentRun = 0

for ($i = 1; $i -le $Iterations; $i++) {
    foreach ($model in $Models) {
        $CurrentRun++
        $PercentComplete = [math]::Round(($CurrentRun / $TotalRuns) * 100, 1)
        
        # Progress bar
        Write-Progress -Activity "Benchmarking $($model.Name)" `
                       -Status "Iteration $i/$Iterations - Overall: $PercentComplete%" `
                       -PercentComplete $PercentComplete
        
        try {
            # Run benchmark
            $output = & $LlamaBench -m $model.Path -t $Threads -r $Repeats -p $PromptTokens -n $GenerateTokens 2>&1
            $outputText = $output -join "`n"
            
            # Parse output - look for tg (token generation) speed
            # Format: | model | size | params | backend | threads | test | t/s |
            # Example: | qwen3 1.7B Q3_K - Small | 948.91 MiB | 2.03 B | CPU | 4 | tg20 | 28.87 ± 1.45 |
            $found = $false
            foreach ($line in $output) {
                $lineStr = $line.ToString()
                # Match pattern: anything with tg followed by speed ± stddev
                if ($lineStr -match "tg\d+\s*\|\s*([\d.]+)\s*±\s*([\d.]+)") {
                    $speed = [double]$Matches[1]
                    [void]$Results[$model.Name].Speeds.Add($speed)
                    $found = $true
                    break
                }
                # Alternative pattern: just numbers at end of line
                elseif ($lineStr -match "\|\s*tg\d+\s*\|\s*([\d.]+)") {
                    $speed = [double]$Matches[1]
                    [void]$Results[$model.Name].Speeds.Add($speed)
                    $found = $true
                    break
                }
            }
            
            if (-not $found) {
                # Debug: show what we got if parsing failed
                if ($i -eq 1) {
                    Write-Host "  Debug - Raw output sample for $($model.Name):" -ForegroundColor DarkGray
                    $output | Select-Object -First 10 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
                }
                $Results[$model.Name].Errors++
            }
        }
        catch {
            $Results[$model.Name].Errors++
            Write-Warning "Error on $($model.Name) iteration $i : $_"
        }
    }
    
    # Periodic status update every 10 iterations
    if ($i % 10 -eq 0) {
        $Elapsed = (Get-Date) - $StartTime
        $EstRemaining = [TimeSpan]::FromSeconds(($Elapsed.TotalSeconds / $CurrentRun) * ($TotalRuns - $CurrentRun))
        Write-Host "  [$i/$Iterations] Elapsed: $($Elapsed.ToString('hh\:mm\:ss')) | ETA: $($EstRemaining.ToString('hh\:mm\:ss'))" -ForegroundColor Gray
    }
}

Write-Progress -Activity "Complete" -Completed

$EndTime = Get-Date
$Duration = $EndTime - $StartTime

# Calculate statistics
function Get-Stats {
    param([System.Collections.ArrayList]$Data)
    
    if ($Data.Count -eq 0) {
        return @{ Mean = 0; StdDev = 0; Min = 0; Max = 0; Median = 0; Count = 0 }
    }
    
    $sorted = $Data | Sort-Object
    $mean = ($Data | Measure-Object -Average).Average
    $min = ($Data | Measure-Object -Minimum).Minimum
    $max = ($Data | Measure-Object -Maximum).Maximum
    $count = $Data.Count
    
    # Median
    $midIndex = [math]::Floor($count / 2)
    if ($count % 2 -eq 0) {
        $median = ($sorted[$midIndex - 1] + $sorted[$midIndex]) / 2
    } else {
        $median = $sorted[$midIndex]
    }
    
    # Standard deviation
    $sumSquares = 0
    foreach ($val in $Data) {
        $sumSquares += [math]::Pow($val - $mean, 2)
    }
    $stdDev = [math]::Sqrt($sumSquares / $count)
    
    # 95th percentile
    $p95Index = [math]::Floor($count * 0.95)
    $p95 = $sorted[[math]::Min($p95Index, $count - 1)]
    
    # 5th percentile  
    $p5Index = [math]::Floor($count * 0.05)
    $p5 = $sorted[$p5Index]
    
    return @{
        Mean = $mean
        StdDev = $stdDev
        Min = $min
        Max = $max
        Median = $median
        P5 = $p5
        P95 = $p95
        Count = $count
    }
}

# Generate report
Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "BENCHMARK RESULTS" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""
Write-Host "Test completed in: $($Duration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
Write-Host "Total iterations per model: $Iterations"
Write-Host ""

# Collect all stats
$AllStats = @{}
foreach ($model in $Models) {
    $AllStats[$model.Name] = Get-Stats -Data $Results[$model.Name].Speeds
}

# Find the fastest model for comparison
$FastestMean = ($AllStats.Values | ForEach-Object { $_.Mean } | Measure-Object -Maximum).Maximum

# Detailed results table
Write-Host "SPEED COMPARISON (tokens/second - higher is better)" -ForegroundColor Yellow
Write-Host "-" * 70

$TableHeader = "{0,-15} {1,10} {2,10} {3,10} {4,10} {5,10} {6,10}" -f "Model", "Mean", "StdDev", "Median", "Min", "Max", "vs Best"
Write-Host $TableHeader -ForegroundColor White
Write-Host "-" * 70

foreach ($model in $Models) {
    $stats = $AllStats[$model.Name]
    $vsBest = if ($stats.Mean -eq $FastestMean) { "FASTEST" } else { 
        "-" + [math]::Round((1 - $stats.Mean / $FastestMean) * 100, 1) + "%" 
    }
    
    $row = "{0,-15} {1,10:F2} {2,10:F2} {3,10:F2} {4,10:F2} {5,10:F2} {6,10}" -f `
        $model.Name, $stats.Mean, $stats.StdDev, $stats.Median, $stats.Min, $stats.Max, $vsBest
    
    if ($stats.Mean -eq $FastestMean) {
        Write-Host $row -ForegroundColor Green
    } else {
        Write-Host $row
    }
}

Write-Host "-" * 70
Write-Host ""

# Percentile analysis
Write-Host "PERCENTILE ANALYSIS" -ForegroundColor Yellow
Write-Host "-" * 70
$PercHeader = "{0,-15} {1,12} {2,12} {3,12} {4,10}" -f "Model", "5th %ile", "Median", "95th %ile", "Samples"
Write-Host $PercHeader -ForegroundColor White
Write-Host "-" * 70

foreach ($model in $Models) {
    $stats = $AllStats[$model.Name]
    $errors = $Results[$model.Name].Errors
    $row = "{0,-15} {1,12:F2} {2,12:F2} {3,12:F2} {4,10}" -f `
        $model.Name, $stats.P5, $stats.Median, $stats.P95, "$($stats.Count)/$Iterations"
    Write-Host $row
}

Write-Host "-" * 70
Write-Host ""

# Speed ranking summary
Write-Host "SPEED RANKING SUMMARY" -ForegroundColor Yellow
Write-Host "-" * 70

$Ranked = @($AllStats.GetEnumerator() | Sort-Object { $_.Value.Mean } -Descending)
$Rank = 1
$FirstMean = if ($Ranked.Count -gt 0 -and $Ranked[0].Value.Mean -gt 0) { $Ranked[0].Value.Mean } else { 1 }

foreach ($entry in $Ranked) {
    $speedDiff = ""
    if ($Rank -gt 1 -and $FirstMean -gt 0 -and $entry.Value.Mean -gt 0) {
        $diffFromFirst = $FirstMean - $entry.Value.Mean
        $diffPercent = ($diffFromFirst / $FirstMean) * 100
        $speedDiff = "($([math]::Round($diffFromFirst, 2)) t/s slower, -$([math]::Round($diffPercent, 1))%)"
    }
    
    $medal = switch ($Rank) { 1 { "🥇" } 2 { "🥈" } 3 { "🥉" } default { "  " } }
    Write-Host "$medal #$Rank $($entry.Key): $([math]::Round($entry.Value.Mean, 2)) ± $([math]::Round($entry.Value.StdDev, 2)) t/s $speedDiff"
    $Rank++
}

Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan

# Export results to CSV
$CsvPath = "benchmark_results_$(Get-Date -Format 'yyyyMMdd_HHmmss').csv"
$CsvData = @()
foreach ($model in $Models) {
    $stats = $AllStats[$model.Name]
    $CsvData += [PSCustomObject]@{
        Model = $model.Name
        Mean_TPS = [math]::Round($stats.Mean, 4)
        StdDev = [math]::Round($stats.StdDev, 4)
        Median = [math]::Round($stats.Median, 4)
        Min = [math]::Round($stats.Min, 4)
        Max = [math]::Round($stats.Max, 4)
        P5 = [math]::Round($stats.P5, 4)
        P95 = [math]::Round($stats.P95, 4)
        Samples = $stats.Count
        Errors = $Results[$model.Name].Errors
    }
}
$CsvData | Export-Csv -Path $CsvPath -NoTypeInformation
Write-Host "Results exported to: $CsvPath" -ForegroundColor Green

# Also save raw data for further analysis
$RawDataPath = "benchmark_raw_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
$RawExport = @{}
foreach ($model in $Models) {
    $RawExport[$model.Name] = $Results[$model.Name].Speeds
}
$RawExport | ConvertTo-Json | Out-File -FilePath $RawDataPath
Write-Host "Raw data exported to: $RawDataPath" -ForegroundColor Green

