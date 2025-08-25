#!/bin/bash

# Test coverage script for Cheungfun project
# Requires cargo-llvm-cov: cargo install cargo-llvm-cov

set -e

echo "🧪 Cheungfun Test Coverage Report"
echo "================================="

# Check if cargo-llvm-cov is installed
if ! command -v cargo-llvm-cov &> /dev/null; then
    echo "❌ cargo-llvm-cov is not installed"
    echo "📦 Installing cargo-llvm-cov..."
    cargo install cargo-llvm-cov
fi

# Create coverage directory
mkdir -p target/coverage

echo "🔍 Running tests with coverage..."

# Run coverage for all packages
cargo llvm-cov clean --workspace
cargo llvm-cov --workspace --all-features --lcov --output-path target/coverage/lcov.info

# Generate HTML report
echo "📊 Generating HTML coverage report..."
cargo llvm-cov --workspace --all-features --html --output-dir target/coverage/html

# Generate summary report
echo "📋 Generating coverage summary..."
cargo llvm-cov --workspace --all-features --summary-only > target/coverage/summary.txt

echo ""
echo "📊 Coverage Summary:"
echo "==================="
cat target/coverage/summary.txt

echo ""
echo "📁 Coverage reports generated:"
echo "  - LCOV format: target/coverage/lcov.info"
echo "  - HTML report: target/coverage/html/index.html"
echo "  - Summary: target/coverage/summary.txt"

# Check if we're in CI environment
if [ "$CI" = "true" ]; then
    echo ""
    echo "🤖 CI Environment detected"
    
    # Extract coverage percentage
    COVERAGE=$(grep -oP 'lines......: \K[0-9.]+' target/coverage/summary.txt | head -1)
    echo "📈 Line coverage: ${COVERAGE}%"
    
    # Set minimum coverage threshold
    THRESHOLD=70
    
    if (( $(echo "$COVERAGE >= $THRESHOLD" | bc -l) )); then
        echo "✅ Coverage threshold met (${COVERAGE}% >= ${THRESHOLD}%)"
        exit 0
    else
        echo "❌ Coverage below threshold (${COVERAGE}% < ${THRESHOLD}%)"
        exit 1
    fi
else
    echo ""
    echo "🌐 Open HTML report:"
    echo "   file://$(pwd)/target/coverage/html/index.html"
fi
