@echo off
setlocal enabledelayedexpansion

echo 🧪 Cheungfun Test Coverage Report
echo =================================

REM Check if cargo-llvm-cov is installed
cargo llvm-cov --version >nul 2>&1
if errorlevel 1 (
    echo ❌ cargo-llvm-cov is not installed
    echo 📦 Installing cargo-llvm-cov...
    cargo install cargo-llvm-cov
    if errorlevel 1 (
        echo ❌ Failed to install cargo-llvm-cov
        exit /b 1
    )
)

REM Create coverage directory
if not exist "target\coverage" mkdir "target\coverage"

echo 🔍 Running tests with coverage...

REM Clean previous coverage data
cargo llvm-cov clean --workspace
if errorlevel 1 (
    echo ❌ Failed to clean coverage data
    exit /b 1
)

REM Run coverage for all packages
cargo llvm-cov --workspace --all-features --lcov --output-path target\coverage\lcov.info
if errorlevel 1 (
    echo ❌ Failed to run coverage tests
    exit /b 1
)

REM Generate HTML report
echo 📊 Generating HTML coverage report...
cargo llvm-cov --workspace --all-features --html --output-dir target\coverage\html
if errorlevel 1 (
    echo ❌ Failed to generate HTML report
    exit /b 1
)

REM Generate summary report
echo 📋 Generating coverage summary...
cargo llvm-cov --workspace --all-features --summary-only > target\coverage\summary.txt
if errorlevel 1 (
    echo ❌ Failed to generate summary
    exit /b 1
)

echo.
echo 📊 Coverage Summary:
echo ===================
type target\coverage\summary.txt

echo.
echo 📁 Coverage reports generated:
echo   - LCOV format: target\coverage\lcov.info
echo   - HTML report: target\coverage\html\index.html
echo   - Summary: target\coverage\summary.txt

REM Check if we're in CI environment
if defined CI (
    echo.
    echo 🤖 CI Environment detected
    
    REM Extract coverage percentage (simplified for Windows)
    for /f "tokens=2 delims=:" %%a in ('findstr "lines" target\coverage\summary.txt') do (
        for /f "tokens=1 delims=%%" %%b in ("%%a") do (
            set COVERAGE=%%b
        )
    )
    
    echo 📈 Line coverage: !COVERAGE!%%
    
    REM Set minimum coverage threshold
    set THRESHOLD=70
    
    REM Simple comparison (note: this is a simplified version)
    if !COVERAGE! GEQ !THRESHOLD! (
        echo ✅ Coverage threshold met ^(!COVERAGE!%% ^>= !THRESHOLD!%%^)
        exit /b 0
    ) else (
        echo ❌ Coverage below threshold ^(!COVERAGE!%% ^< !THRESHOLD!%%^)
        exit /b 1
    )
) else (
    echo.
    echo 🌐 Open HTML report:
    echo    file:///%CD:\=/%/target/coverage/html/index.html
    
    REM Try to open the report in the default browser
    start "" "target\coverage\html\index.html" 2>nul
)

endlocal
