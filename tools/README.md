# Cheungfun Tools

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Development tools for Cheungfun**

Cheungfun-tools provides development utilities and test runners for the Cheungfun RAG framework. It includes comprehensive test orchestration, performance benchmarking, and development workflow automation.

> **⚠️ Learning Project Disclaimer**: This is a personal learning project for exploring RAG architecture design in Rust. While functionally complete, it is still under development and **not recommended for production use**.

## 🚀 Features

### 🧪 Test Runner
- **Categorized Testing**: Unit, integration, performance, and specialized test suites
- **Package-Specific Tests**: Run tests for individual crates or the entire workspace
- **Feature-Aware Testing**: Automatically enable required features for test execution
- **Performance Benchmarking**: Dedicated performance test runner with release mode optimization
- **Comprehensive Reporting**: Detailed test results with timing and failure analysis

### 🎯 Test Categories
- **Unit Tests**: Core functionality testing for individual modules
- **Integration Tests**: Cross-module integration and system testing
- **Performance Tests**: SIMD, HNSW, and storage performance benchmarks
- **Storage Tests**: Database and persistence layer testing
- **Memory Tests**: Conversation memory and caching system testing
- **Config Tests**: Configuration management and serialization testing

## 📦 Installation

This crate is primarily used as a development dependency within the Cheungfun workspace:

```toml
[dev-dependencies]
cheungfun-tools = { path = "../tools" }
```

## 🚀 Usage

### Test Runner

Run the test runner from the workspace root:

```bash
# Run default test suite (unit + basic integration)
cargo run --package cheungfun-tools --bin run_tests

# Run specific test categories
cargo run --package cheungfun-tools --bin run_tests unit
cargo run --package cheungfun-tools --bin run_tests integration
cargo run --package cheungfun-tools --bin run_tests performance

# Run specialized test suites
cargo run --package cheungfun-tools --bin run_tests storage
cargo run --package cheungfun-tools --bin run_tests memory
cargo run --package cheungfun-tools --bin run_tests config

# Run complete test suite
cargo run --package cheungfun-tools --bin run_tests all

# Show help
cargo run --package cheungfun-tools --bin run_tests help
```

### Test Categories

#### Unit Tests
```bash
cargo run --package cheungfun-tools --bin run_tests unit
```
- Core functionality testing
- Individual module validation
- Fast execution for development workflow

#### Integration Tests
```bash
cargo run --package cheungfun-tools --bin run_tests integration
```
- Cross-module integration testing
- System-level functionality validation
- Database and storage integration

#### Performance Tests
```bash
cargo run --package cheungfun-tools --bin run_tests performance
```
- SIMD vector operations benchmarking
- HNSW search performance testing
- Storage system performance validation
- Runs in release mode for accurate measurements

#### Specialized Tests
```bash
# Storage-specific testing
cargo run --package cheungfun-tools --bin run_tests storage

# Memory system testing
cargo run --package cheungfun-tools --bin run_tests memory

# Configuration system testing
cargo run --package cheungfun-tools --bin run_tests config
```

## 🏗️ Architecture

### Test Suite Configuration

```rust
use cheungfun_tools::test_runner::*;

let suite = TestSuite {
    name: "Core Unit Tests",
    package: Some("cheungfun-core"),
    features: vec!["storage", "simd"],
    test_type: TestType::Unit,
    description: "Core functionality testing",
};
```

### Test Types

- **`TestType::Unit`**: Library tests only (`--lib`)
- **`TestType::Integration`**: Integration tests (`--test *integration*`)
- **`TestType::Performance`**: Performance tests in release mode
- **`TestType::All`**: Complete test suite for a package

### Test Execution Flow

```text
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Test Suite    │    │   Cargo Test    │    │   Results       │
│   Configuration │ -> │   Execution     │ -> │   Analysis      │
│                 │    │                 │    │                 │
│ - Package       │    │ - Features      │    │ - Timing        │
│ - Features      │    │ - Test Type     │    │ - Pass/Fail     │
│ - Test Type     │    │ - Filters       │    │ - Summary       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📊 Test Suites

### Default Test Suite
- Core unit tests
- Query unit tests  
- Indexing unit tests
- Basic integration validation

### Complete Test Suite
1. **Phase 1**: All unit tests
2. **Phase 2**: All integration tests
3. **Phase 3**: All performance tests

### Package-Specific Suites
- **cheungfun-core**: Core traits and types
- **cheungfun-query**: Query engine and memory
- **cheungfun-indexing**: Document processing
- **cheungfun-integrations**: External services
- **cheungfun-agents**: Agent framework
- **cheungfun-multimodal**: Multimodal processing

## 🔧 Development Workflow

### Quick Development Testing
```bash
# Fast unit tests during development
cargo run --package cheungfun-tools --bin run_tests unit
```

### Pre-Commit Testing
```bash
# Comprehensive testing before commits
cargo run --package cheungfun-tools --bin run_tests integration
```

### Performance Validation
```bash
# Performance regression testing
cargo run --package cheungfun-tools --bin run_tests performance
```

### Release Testing
```bash
# Complete test suite for releases
cargo run --package cheungfun-tools --bin run_tests all
```

## 📈 Performance Testing

Performance tests are automatically run in release mode with optimizations enabled:

- **SIMD Operations**: Vector computation benchmarks
- **HNSW Search**: Approximate nearest neighbor performance
- **Storage Systems**: Database operation benchmarks
- **Memory Management**: Caching and conversation memory performance

## 🧪 Test Output

```
🧪 Cheungfun Test Runner
========================
📦 Running: Core Unit Tests
   Description: Core traits, types, and utilities
   ✅ PASSED (2.34s)

📦 Running: Query Unit Tests
   Description: Query engine and memory system
   ✅ PASSED (1.87s)

📊 Test Summary
===============
Total suites: 2
Passed: 2
Failed: 0
Duration: 4.21s

🎉 All tests passed!
```

## 🔗 Related Crates

This tool is designed to work with all Cheungfun crates:

- **[cheungfun-core](../cheungfun-core)**: Core traits and types
- **[cheungfun-indexing](../cheungfun-indexing)**: Document processing
- **[cheungfun-query](../cheungfun-query)**: Query processing
- **[cheungfun-agents](../cheungfun-agents)**: Agent framework
- **[cheungfun-integrations](../cheungfun-integrations)**: External integrations
- **[cheungfun-multimodal](../cheungfun-multimodal)**: Multimodal processing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
