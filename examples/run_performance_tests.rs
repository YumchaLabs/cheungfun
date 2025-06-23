//! Unified performance testing entry point for Cheungfun
//!
//! This script provides a convenient way to run all performance tests
//! with appropriate feature flags and configurations.
//!
//! ## Usage Examples:
//!
//! Run all tests with default features:
//! ```bash
//! cargo run --bin run_performance_tests
//! ```
//!
//! Run with SIMD acceleration:
//! ```bash
//! cargo run --bin run_performance_tests --features simd
//! ```
//!
//! Run with all performance features:
//! ```bash
//! cargo run --bin run_performance_tests --features performance
//! ```
//!
//! Run specific test categories:
//! ```bash
//! cargo run --bin run_performance_tests -- --embedders --vector-stores
//! ```

use std::env;
use std::process::{Command, Stdio};
use std::time::Instant;

/// Available test categories
#[derive(Debug, Clone)]
struct TestConfig {
    run_embedders: bool,
    run_vector_stores: bool,
    run_end_to_end: bool,
    run_comprehensive: bool,
    use_release: bool,
    features: Vec<String>,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            run_embedders: true,
            run_vector_stores: true,
            run_end_to_end: true,
            run_comprehensive: true,
            use_release: true,
            features: vec!["simd".to_string()], // Default to SIMD for better performance
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Cheungfun Performance Testing Suite");
    println!("======================================");
    
    let config = parse_args()?;
    print_config(&config);
    
    let start_time = Instant::now();
    let mut results = Vec::new();
    
    // Run comprehensive performance benchmark (main test)
    if config.run_comprehensive {
        println!("\n📊 Running Comprehensive Performance Benchmark");
        println!("==============================================");
        
        let result = run_test("performance_benchmark", &config)?;
        results.push(("Comprehensive Benchmark", result));
    }
    
    // Run specific component tests
    if config.run_embedders {
        println!("\n🔥 Running Embedder Benchmarks");
        println!("==============================");
        
        let result = run_test("embedder_benchmark", &config)?;
        results.push(("Embedder Benchmark", result));
    }
    
    if config.run_vector_stores {
        println!("\n🗄️  Running Vector Store Benchmarks");
        println!("===================================");
        
        let result = run_test("vector_store_benchmark", &config)?;
        results.push(("Vector Store Benchmark", result));
    }
    
    if config.run_end_to_end {
        println!("\n🔄 Running End-to-End Benchmarks");
        println!("================================");
        
        let result = run_test("end_to_end_benchmark", &config)?;
        results.push(("End-to-End Benchmark", result));
    }
    
    // Print summary
    let total_time = start_time.elapsed();
    print_summary(&results, total_time);
    
    Ok(())
}

/// Parse command line arguments
fn parse_args() -> Result<TestConfig, Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let mut config = TestConfig::default();
    
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--embedders" => {
                config.run_embedders = true;
                config.run_vector_stores = false;
                config.run_end_to_end = false;
                config.run_comprehensive = false;
            }
            "--vector-stores" => {
                config.run_vector_stores = true;
                config.run_embedders = false;
                config.run_end_to_end = false;
                config.run_comprehensive = false;
            }
            "--end-to-end" => {
                config.run_end_to_end = true;
                config.run_embedders = false;
                config.run_vector_stores = false;
                config.run_comprehensive = false;
            }
            "--comprehensive" => {
                config.run_comprehensive = true;
                config.run_embedders = false;
                config.run_vector_stores = false;
                config.run_end_to_end = false;
            }
            "--debug" => {
                config.use_release = false;
            }
            "--features" => {
                if i + 1 < args.len() {
                    config.features = args[i + 1].split(',').map(|s| s.trim().to_string()).collect();
                    i += 1;
                }
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                print_help();
                std::process::exit(1);
            }
        }
        i += 1;
    }
    
    Ok(config)
}

/// Print configuration
fn print_config(config: &TestConfig) {
    println!("Configuration:");
    println!("  Embedders: {}", config.run_embedders);
    println!("  Vector Stores: {}", config.run_vector_stores);
    println!("  End-to-End: {}", config.run_end_to_end);
    println!("  Comprehensive: {}", config.run_comprehensive);
    println!("  Release Mode: {}", config.use_release);
    println!("  Features: {:?}", config.features);
    
    // Show enabled features
    println!("\nEnabled Rust features:");
    #[cfg(feature = "simd")]
    println!("  ✅ SIMD acceleration");
    #[cfg(not(feature = "simd"))]
    println!("  ❌ SIMD acceleration");
    
    #[cfg(feature = "optimized-memory")]
    println!("  ✅ Optimized memory stores");
    #[cfg(not(feature = "optimized-memory"))]
    println!("  ❌ Optimized memory stores");
    
    #[cfg(feature = "hnsw")]
    println!("  ✅ HNSW approximate search");
    #[cfg(not(feature = "hnsw"))]
    println!("  ❌ HNSW approximate search");
    
    #[cfg(feature = "performance")]
    println!("  ✅ Performance bundle");
    #[cfg(not(feature = "performance"))]
    println!("  ❌ Performance bundle");
}

/// Run a specific test
fn run_test(test_name: &str, config: &TestConfig) -> Result<bool, Box<dyn std::error::Error>> {
    let mut cmd = Command::new("cargo");
    cmd.arg("run")
       .arg("--bin")
       .arg(test_name);
    
    if config.use_release {
        cmd.arg("--release");
    }
    
    if !config.features.is_empty() {
        cmd.arg("--features");
        cmd.arg(config.features.join(","));
    }
    
    cmd.stdout(Stdio::inherit())
       .stderr(Stdio::inherit());
    
    println!("Running: {:?}", cmd);
    
    let start = Instant::now();
    let status = cmd.status()?;
    let duration = start.elapsed();
    
    let success = status.success();
    let status_icon = if success { "✅" } else { "❌" };
    
    println!("{} {} completed in {:?}", status_icon, test_name, duration);
    
    Ok(success)
}

/// Print test summary
fn print_summary(results: &[(&str, bool)], total_time: std::time::Duration) {
    println!("\n📈 Performance Testing Summary");
    println!("=============================");
    
    let successful = results.iter().filter(|(_, success)| *success).count();
    let total = results.len();
    
    println!("Total tests: {}", total);
    println!("Successful: {}", successful);
    println!("Failed: {}", total - successful);
    println!("Total time: {:?}", total_time);
    
    println!("\nDetailed results:");
    for (name, success) in results {
        let status = if *success { "✅ PASS" } else { "❌ FAIL" };
        println!("  {} {}", status, name);
    }
    
    if successful == total {
        println!("\n🎉 All performance tests completed successfully!");
    } else {
        println!("\n⚠️  Some performance tests failed. Check the output above for details.");
    }
}

/// Print help message
fn print_help() {
    println!("Cheungfun Performance Testing Suite");
    println!();
    println!("USAGE:");
    println!("    cargo run --bin run_performance_tests [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    --embedders        Run only embedder benchmarks");
    println!("    --vector-stores    Run only vector store benchmarks");
    println!("    --end-to-end       Run only end-to-end benchmarks");
    println!("    --comprehensive    Run only comprehensive benchmark (default: all)");
    println!("    --debug            Use debug build instead of release");
    println!("    --features <list>  Comma-separated list of features to enable");
    println!("    --help, -h         Show this help message");
    println!();
    println!("EXAMPLES:");
    println!("    cargo run --bin run_performance_tests");
    println!("    cargo run --bin run_performance_tests --features simd,hnsw");
    println!("    cargo run --bin run_performance_tests --embedders --debug");
}
