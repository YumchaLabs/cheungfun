//! Shared utilities for Cheungfun examples
//!
//! This module contains common utilities, frameworks, and helpers used across
//! multiple examples in the Cheungfun examples collection.

/// Performance benchmarking framework
pub mod benchmark_framework;

/// Report generation utilities
pub mod report_generator;

/// Common utilities and helpers
pub mod common;

// Re-export commonly used items
pub use benchmark_framework::{
    BenchmarkConfig, BenchmarkRunner, PerformanceMetrics, 
    run_benchmark, format_metrics, generate_csv_report, compare_metrics
};

pub use report_generator::ReportGenerator;
pub use common::*;
