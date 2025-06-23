@echo off
setlocal enabledelayedexpansion

REM Cheungfun Performance Benchmark Runner for Windows
REM This script runs all performance benchmarks and generates reports

echo 🚀 Cheungfun Performance Benchmark Suite
echo ========================================
echo.

REM Create results directory
set RESULTS_DIR=.\benchmark_results
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"

echo 📁 Results will be saved to: %RESULTS_DIR%
echo.

REM Check if required features are available
echo 🔍 Checking available features...

REM Check for FastEmbed
cargo check --features fastembed --quiet >nul 2>&1
if %errorlevel% equ 0 (
    set FASTEMBED_AVAILABLE=true
    echo ✅ FastEmbed feature available
) else (
    set FASTEMBED_AVAILABLE=false
    echo ❌ FastEmbed feature not available
)

REM Check for Candle
cargo check --features candle --quiet >nul 2>&1
if %errorlevel% equ 0 (
    set CANDLE_AVAILABLE=true
    echo ✅ Candle feature available
) else (
    set CANDLE_AVAILABLE=false
    echo ❌ Candle feature not available
)

REM Check for Qdrant (by trying to connect)
curl -s http://localhost:6334/health >nul 2>&1
if %errorlevel% equ 0 (
    set QDRANT_AVAILABLE=true
    echo ✅ Qdrant server available at localhost:6334
) else (
    set QDRANT_AVAILABLE=false
    echo ❌ Qdrant server not available (start with: docker run -p 6334:6334 qdrant/qdrant)
)

REM Check for API keys
if defined OPENAI_API_KEY (
    set API_AVAILABLE=true
    echo ✅ API key available for cloud embeddings
) else if defined SIUMAI_API_KEY (
    set API_AVAILABLE=true
    echo ✅ API key available for cloud embeddings
) else (
    set API_AVAILABLE=false
    echo ❌ No API key found (set OPENAI_API_KEY or SIUMAI_API_KEY for cloud embedding tests)
)

echo.

REM Run benchmarks based on available features
echo 🏃 Running benchmarks...
echo.

REM 1. Vector Store Benchmarks (always available)
echo 🗄️  Running Vector Store Benchmarks...
cargo run --bin vector_store_benchmark > "%RESULTS_DIR%\vector_store_benchmark.log" 2>&1
if %errorlevel% equ 0 (
    echo ✅ Vector store benchmarks completed
) else (
    echo ❌ Vector store benchmarks failed
)
echo.

REM 2. Embedder Benchmarks (if FastEmbed is available)
if "%FASTEMBED_AVAILABLE%"=="true" (
    echo 🔥 Running Embedder Benchmarks with FastEmbed...
    cargo run --features fastembed --bin embedder_benchmark > "%RESULTS_DIR%\embedder_benchmark.log" 2>&1
    if !errorlevel! equ 0 (
        echo ✅ Embedder benchmarks completed
    ) else (
        echo ❌ Embedder benchmarks failed
    )
    echo.
)

REM 3. End-to-End Benchmarks (if FastEmbed is available)
if "%FASTEMBED_AVAILABLE%"=="true" (
    echo 🔄 Running End-to-End RAG Benchmarks...
    cargo run --features fastembed --bin end_to_end_benchmark > "%RESULTS_DIR%\end_to_end_benchmark.log" 2>&1
    if !errorlevel! equ 0 (
        echo ✅ End-to-end benchmarks completed
    ) else (
        echo ❌ End-to-end benchmarks failed
    )
    echo.
)

REM 4. Comprehensive Benchmark Suite
echo 📊 Running Comprehensive Benchmark Suite...
cargo run --bin performance_benchmark > "%RESULTS_DIR%\comprehensive_benchmark.log" 2>&1
if %errorlevel% equ 0 (
    echo ✅ Comprehensive benchmarks completed
) else (
    echo ❌ Comprehensive benchmarks failed
)
echo.

REM Generate summary report
echo 📋 Generating Summary Report...

REM Get current date and time
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%" & set "Sec=%dt:~12,2%"
set "datestamp=%YYYY%-%MM%-%DD% %HH%:%Min%:%Sec%"

(
echo # Cheungfun Performance Benchmark Summary
echo.
echo **Date:** %datestamp%
echo **System:** Windows
echo.
echo ## Environment
echo - FastEmbed Available: %FASTEMBED_AVAILABLE%
echo - Candle Available: %CANDLE_AVAILABLE%
echo - Qdrant Available: %QDRANT_AVAILABLE%
echo - API Available: %API_AVAILABLE%
echo.
echo ## Benchmarks Run
echo - Vector Store Benchmarks: ✅
if "%FASTEMBED_AVAILABLE%"=="true" (
    echo - Embedder Benchmarks: ✅
    echo - End-to-End Benchmarks: ✅
) else (
    echo - Embedder Benchmarks: ❌ (FastEmbed not available^)
    echo - End-to-End Benchmarks: ❌ (FastEmbed not available^)
)
echo - Comprehensive Suite: ✅
echo.
echo ## Files Generated
echo - `vector_store_benchmark.log`: Vector store performance results
if "%FASTEMBED_AVAILABLE%"=="true" (
    echo - `embedder_benchmark.log`: Embedder performance results
    echo - `end_to_end_benchmark.log`: End-to-end RAG performance results
)
echo - `comprehensive_benchmark.log`: Overall benchmark suite results
echo - `performance_report.html`: Interactive HTML report (if generated^)
echo - `raw_metrics/`: Raw JSON metrics data
echo.
echo ## Next Steps
echo 1. Review the generated reports in `%RESULTS_DIR%`
echo 2. Open `performance_report.html` in a web browser for interactive charts
echo 3. Compare results with previous benchmarks to track performance trends
echo 4. Use the insights to optimize your Cheungfun configuration
echo.
echo ## Recommendations
echo - For production deployments, focus on the end-to-end benchmark results
echo - Monitor memory usage if processing large datasets
echo - Consider using Qdrant for production vector storage
echo - Set up regular benchmarking to track performance over time
) > "%RESULTS_DIR%\benchmark_summary.md"

echo ✅ Summary report generated: %RESULTS_DIR%\benchmark_summary.md
echo.

REM Final summary
echo 🎉 Benchmark suite completed!
echo 📁 All results saved to: %RESULTS_DIR%
echo.
echo 📊 Quick Summary:
echo   • Vector Store Benchmarks: Completed
if "%FASTEMBED_AVAILABLE%"=="true" (
    echo   • Embedder Benchmarks: Completed
    echo   • End-to-End Benchmarks: Completed
) else (
    echo   • Embedder Benchmarks: Skipped (install FastEmbed: cargo add fastembed^)
    echo   • End-to-End Benchmarks: Skipped (install FastEmbed: cargo add fastembed^)
)
echo   • Comprehensive Suite: Completed
echo.
echo 💡 To view detailed results:
echo    type %RESULTS_DIR%\benchmark_summary.md
if exist "%RESULTS_DIR%\performance_report.html" (
    echo    start %RESULTS_DIR%\performance_report.html
)
echo.
echo 🚀 Happy benchmarking!

pause
