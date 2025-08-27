//! Examples for the Cheungfun RAG framework.
//!
//! This crate contains practical examples demonstrating how to use
//! the Cheungfun framework for building RAG applications.
//!
//! # Cheungfun Examples Library
//!
//! This library provides shared utilities and frameworks for Cheungfun examples.
//!
//! ## Directory Structure
//!
//! - `01_getting_started/` - Basic usage examples for beginners
//! - `02_core_components/` - Individual component demonstrations
//! - `03_advanced_features/` - Advanced RAG functionality
//! - `04_integrations/` - External system integrations
//! - `05_performance/` - Performance testing and optimization
//! - `06_production/` - Production-ready implementations
//! - `07_use_cases/` - Real-world application examples
//! - `shared/` - Common utilities and frameworks
//!
//! ## Quick Start
//!
//! ```bash
//! # Start with the basics
//! cargo run --bin hello_world
//! cargo run --bin basic_indexing
//! cargo run --bin basic_querying
//!
//! # Explore components
//! cargo run --features fastembed --bin embedder_demo
//! cargo run --bin vector_store_demo
//!
//! # Performance testing
//! cargo run --bin vector_store_benchmark
//! ```

#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

// Re-export shared modules from their new locations
pub use crate::shared::*;

/// Shared utilities and frameworks
pub mod shared {
    /// Performance benchmarking framework
    pub mod benchmark_framework;
    /// Common utilities used across examples
    pub mod common;
    /// Report generation utilities
    pub mod report_generator;
}
