//! Test runner for Cheungfun project.

use cheungfun_tools::test_runner::*;
use std::env;

fn main() {
    println!("🧪 Cheungfun Test Runner");
    println!("========================");

    let args: Vec<String> = env::args().collect();

    if args.len() > 1 {
        match args[1].as_str() {
            "unit" => run_unit_tests(),
            "integration" => run_integration_tests(),
            "performance" => run_performance_tests(),
            "storage" => run_storage_tests(),
            "memory" => run_memory_tests(),
            "config" => run_config_tests(),
            "all" => run_all_tests(),
            "help" | "--help" | "-h" => print_help(),
            _ => {
                println!("❌ Unknown test category: {}", args[1]);
                print_help();
                std::process::exit(1);
            }
        }
    } else {
        run_default_tests();
    }
}

fn run_default_tests() {
    println!("🚀 Running default test suite (unit + basic integration)...\n");
    let test_suites = get_default_test_suites();
    run_test_suites(&test_suites);
}

fn run_unit_tests() {
    println!("🔬 Running unit tests...\n");
    let test_suites = get_unit_test_suites();
    run_test_suites(&test_suites);
}

fn run_integration_tests() {
    println!("🔗 Running integration tests...\n");
    let test_suites = get_integration_test_suites();
    run_test_suites(&test_suites);
}

fn run_performance_tests() {
    println!("⚡ Running performance tests...\n");
    let test_suites = get_performance_test_suites();
    run_test_suites(&test_suites);
}

fn run_storage_tests() {
    println!("🗄️  Running storage-specific tests...\n");
    let test_suites = get_storage_test_suites();
    run_test_suites(&test_suites);
}

fn run_memory_tests() {
    println!("🧠 Running memory-specific tests...\n");
    let test_suites = get_memory_test_suites();
    run_test_suites(&test_suites);
}

fn run_config_tests() {
    println!("⚙️  Running configuration-specific tests...\n");
    let test_suites = get_config_test_suites();
    run_test_suites(&test_suites);
}

fn run_all_tests() {
    println!("🎯 Running complete test suite...\n");

    println!("Phase 1: Unit Tests");
    run_unit_tests();

    println!("\nPhase 2: Integration Tests");
    run_integration_tests();

    println!("\nPhase 3: Performance Tests");
    run_performance_tests();
}
