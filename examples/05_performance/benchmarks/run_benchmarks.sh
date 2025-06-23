#!/bin/bash

# Cheungfun Performance Benchmark Runner
# This script runs all performance benchmarks and generates reports

set -e

echo "🚀 Cheungfun Performance Benchmark Suite"
echo "========================================"
echo ""

# Create results directory
RESULTS_DIR="./benchmark_results"
mkdir -p "$RESULTS_DIR"

echo "📁 Results will be saved to: $RESULTS_DIR"
echo ""

# Check if required features are available
echo "🔍 Checking available features..."

# Check for FastEmbed
if cargo check --features fastembed --quiet 2>/dev/null; then
    FASTEMBED_AVAILABLE=true
    echo "✅ FastEmbed feature available"
else
    FASTEMBED_AVAILABLE=false
    echo "❌ FastEmbed feature not available"
fi

# Check for Candle
if cargo check --features candle --quiet 2>/dev/null; then
    CANDLE_AVAILABLE=true
    echo "✅ Candle feature available"
else
    CANDLE_AVAILABLE=false
    echo "❌ Candle feature not available"
fi

# Check for Qdrant (by trying to connect)
if curl -s http://localhost:6334/health >/dev/null 2>&1; then
    QDRANT_AVAILABLE=true
    echo "✅ Qdrant server available at localhost:6334"
else
    QDRANT_AVAILABLE=false
    echo "❌ Qdrant server not available (start with: docker run -p 6334:6334 qdrant/qdrant)"
fi

# Check for API keys
if [ -n "$OPENAI_API_KEY" ] || [ -n "$SIUMAI_API_KEY" ]; then
    API_AVAILABLE=true
    echo "✅ API key available for cloud embeddings"
else
    API_AVAILABLE=false
    echo "❌ No API key found (set OPENAI_API_KEY or SIUMAI_API_KEY for cloud embedding tests)"
fi

echo ""

# Run benchmarks based on available features
echo "🏃 Running benchmarks..."
echo ""

# 1. Vector Store Benchmarks (always available)
echo "🗄️  Running Vector Store Benchmarks..."
if cargo run --bin vector_store_benchmark 2>&1 | tee "$RESULTS_DIR/vector_store_benchmark.log"; then
    echo "✅ Vector store benchmarks completed"
else
    echo "❌ Vector store benchmarks failed"
fi
echo ""

# 2. Embedder Benchmarks (if FastEmbed is available)
if [ "$FASTEMBED_AVAILABLE" = true ]; then
    echo "🔥 Running Embedder Benchmarks with FastEmbed..."
    if cargo run --features fastembed --bin embedder_benchmark 2>&1 | tee "$RESULTS_DIR/embedder_benchmark.log"; then
        echo "✅ Embedder benchmarks completed"
    else
        echo "❌ Embedder benchmarks failed"
    fi
    echo ""
fi

# 3. End-to-End Benchmarks (if FastEmbed is available)
if [ "$FASTEMBED_AVAILABLE" = true ]; then
    echo "🔄 Running End-to-End RAG Benchmarks..."
    if cargo run --features fastembed --bin end_to_end_benchmark 2>&1 | tee "$RESULTS_DIR/end_to_end_benchmark.log"; then
        echo "✅ End-to-end benchmarks completed"
    else
        echo "❌ End-to-end benchmarks failed"
    fi
    echo ""
fi

# 4. Comprehensive Benchmark Suite
echo "📊 Running Comprehensive Benchmark Suite..."
if cargo run --bin performance_benchmark 2>&1 | tee "$RESULTS_DIR/comprehensive_benchmark.log"; then
    echo "✅ Comprehensive benchmarks completed"
else
    echo "❌ Comprehensive benchmarks failed"
fi
echo ""

# Generate summary report
echo "📋 Generating Summary Report..."
cat > "$RESULTS_DIR/benchmark_summary.md" << EOF
# Cheungfun Performance Benchmark Summary

**Date:** $(date)
**System:** $(uname -a)

## Environment
- FastEmbed Available: $FASTEMBED_AVAILABLE
- Candle Available: $CANDLE_AVAILABLE
- Qdrant Available: $QDRANT_AVAILABLE
- API Available: $API_AVAILABLE

## Benchmarks Run
- Vector Store Benchmarks: ✅
- Embedder Benchmarks: $([ "$FASTEMBED_AVAILABLE" = true ] && echo "✅" || echo "❌ (FastEmbed not available)")
- End-to-End Benchmarks: $([ "$FASTEMBED_AVAILABLE" = true ] && echo "✅" || echo "❌ (FastEmbed not available)")
- Comprehensive Suite: ✅

## Files Generated
- \`vector_store_benchmark.log\`: Vector store performance results
$([ "$FASTEMBED_AVAILABLE" = true ] && echo "- \`embedder_benchmark.log\`: Embedder performance results")
$([ "$FASTEMBED_AVAILABLE" = true ] && echo "- \`end_to_end_benchmark.log\`: End-to-end RAG performance results")
- \`comprehensive_benchmark.log\`: Overall benchmark suite results
- \`performance_report.html\`: Interactive HTML report (if generated)
- \`raw_metrics/\`: Raw JSON metrics data

## Next Steps
1. Review the generated reports in \`$RESULTS_DIR\`
2. Open \`performance_report.html\` in a web browser for interactive charts
3. Compare results with previous benchmarks to track performance trends
4. Use the insights to optimize your Cheungfun configuration

## Recommendations
- For production deployments, focus on the end-to-end benchmark results
- Monitor memory usage if processing large datasets
- Consider using Qdrant for production vector storage
- Set up regular benchmarking to track performance over time
EOF

echo "✅ Summary report generated: $RESULTS_DIR/benchmark_summary.md"
echo ""

# Final summary
echo "🎉 Benchmark suite completed!"
echo "📁 All results saved to: $RESULTS_DIR"
echo ""
echo "📊 Quick Summary:"
echo "  • Vector Store Benchmarks: Completed"
if [ "$FASTEMBED_AVAILABLE" = true ]; then
    echo "  • Embedder Benchmarks: Completed"
    echo "  • End-to-End Benchmarks: Completed"
else
    echo "  • Embedder Benchmarks: Skipped (install FastEmbed: cargo add fastembed)"
    echo "  • End-to-End Benchmarks: Skipped (install FastEmbed: cargo add fastembed)"
fi
echo "  • Comprehensive Suite: Completed"
echo ""
echo "💡 To view detailed results:"
echo "   cat $RESULTS_DIR/benchmark_summary.md"
if [ -f "$RESULTS_DIR/performance_report.html" ]; then
    echo "   open $RESULTS_DIR/performance_report.html"
fi
echo ""
echo "🚀 Happy benchmarking!"
