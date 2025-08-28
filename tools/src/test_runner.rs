//! Test runner implementation for Cheungfun project.

use std::process::{Command, Stdio};
use std::time::Instant;

/// Test suite configuration.
#[derive(Debug, Clone)]
pub struct TestSuite {
    /// Name of the test suite.
    pub name: &'static str,
    /// Package to test (optional).
    pub package: Option<&'static str>,
    /// Features to enable.
    pub features: Vec<&'static str>,
    /// Type of test.
    pub test_type: TestType,
    /// Description of the test suite.
    pub description: &'static str,
}

/// Type of test to run.
#[derive(Debug, Clone)]
pub enum TestType {
    /// Unit tests only.
    Unit,
    /// Integration tests only.
    Integration,
    /// Performance tests only.
    Performance,
    /// All tests.
    All,
}

/// Run a collection of test suites.
pub fn run_test_suites(test_suites: &[TestSuite]) {
    let mut total_passed = 0;
    let mut total_failed = 0;
    let start_time = Instant::now();

    for suite in test_suites {
        println!("📦 Running: {}", suite.name);
        println!("   Description: {}", suite.description);

        let suite_start = Instant::now();
        let result = run_test_suite(suite);
        let suite_duration = suite_start.elapsed();

        match result {
            Ok(()) => {
                println!("   ✅ PASSED ({:.2}s)", suite_duration.as_secs_f64());
                total_passed += 1;
            }
            Err(e) => {
                println!("   ❌ FAILED ({:.2}s): {}", suite_duration.as_secs_f64(), e);
                total_failed += 1;
            }
        }
        println!();
    }

    let total_duration = start_time.elapsed();
    println!("📊 Test Summary");
    println!("===============");
    println!("Total suites: {}", test_suites.len());
    println!("Passed: {total_passed}");
    println!("Failed: {total_failed}");
    println!("Duration: {:.2}s", total_duration.as_secs_f64());

    if total_failed > 0 {
        println!("\n❌ Some tests failed!");
        std::process::exit(1);
    } else {
        println!("\n🎉 All tests passed!");
    }
}

/// Run a single test suite.
///
/// # Errors
///
/// Returns an error if the test execution fails or if the tests fail.
pub fn run_test_suite(suite: &TestSuite) -> Result<(), String> {
    let mut cmd = Command::new("cargo");
    cmd.arg("test");

    // Add package filter if specified
    if let Some(package) = suite.package {
        cmd.arg("--package").arg(package);
    }

    // Add features if specified
    if !suite.features.is_empty() {
        cmd.arg("--features").arg(suite.features.join(","));
    }

    // Add test type specific flags
    match suite.test_type {
        TestType::Unit => {
            cmd.arg("--lib");
        }
        TestType::Integration => {
            cmd.arg("--test").arg("*integration*");
        }
        TestType::Performance => {
            cmd.arg("--test").arg("*performance*");
            cmd.arg("--release"); // Performance tests should run in release mode
        }
        TestType::All => {
            // Run all tests for the package
        }
    }

    // Configure output
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    // Execute the command
    let output = cmd
        .output()
        .map_err(|e| format!("Failed to execute cargo test: {e}"))?;

    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        Err(format!(
            "Test failed:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        ))
    }
}

/// Get default test suites.
#[must_use]
pub fn get_default_test_suites() -> Vec<TestSuite> {
    vec![
        TestSuite {
            name: "Core Unit Tests",
            package: Some("cheungfun-core"),
            features: vec![],
            test_type: TestType::Unit,
            description: "Core traits, types, and utilities",
        },
        TestSuite {
            name: "Query Unit Tests",
            package: Some("cheungfun-query"),
            features: vec![],
            test_type: TestType::Unit,
            description: "Query engine and memory system",
        },
        TestSuite {
            name: "Indexing Unit Tests",
            package: Some("cheungfun-indexing"),
            features: vec![],
            test_type: TestType::Unit,
            description: "Document indexing and processing",
        },
    ]
}

/// Get unit test suites.
#[must_use]
pub fn get_unit_test_suites() -> Vec<TestSuite> {
    vec![
        TestSuite {
            name: "Core Unit Tests",
            package: Some("cheungfun-core"),
            features: vec![],
            test_type: TestType::Unit,
            description: "Core functionality",
        },
        TestSuite {
            name: "Query Unit Tests",
            package: Some("cheungfun-query"),
            features: vec![],
            test_type: TestType::Unit,
            description: "Query and memory systems",
        },
        TestSuite {
            name: "Indexing Unit Tests",
            package: Some("cheungfun-indexing"),
            features: vec![],
            test_type: TestType::Unit,
            description: "Document processing",
        },
        TestSuite {
            name: "Integrations Unit Tests",
            package: Some("cheungfun-integrations"),
            features: vec![],
            test_type: TestType::Unit,
            description: "External integrations",
        },
    ]
}

/// Get integration test suites.
#[must_use]
pub fn get_integration_test_suites() -> Vec<TestSuite> {
    vec![
        TestSuite {
            name: "Storage Integration Tests",
            package: Some("cheungfun-integrations"),
            features: vec!["storage"],
            test_type: TestType::Integration,
            description: "Database and storage systems",
        },
        TestSuite {
            name: "Memory Integration Tests",
            package: Some("cheungfun-query"),
            features: vec![],
            test_type: TestType::Integration,
            description: "Conversation memory management",
        },
        TestSuite {
            name: "Config Integration Tests",
            package: Some("cheungfun-core"),
            features: vec![],
            test_type: TestType::Integration,
            description: "Configuration management",
        },
    ]
}

/// Get performance test suites.
#[must_use]
pub fn get_performance_test_suites() -> Vec<TestSuite> {
    vec![
        TestSuite {
            name: "SIMD Performance Tests",
            package: Some("cheungfun-integrations"),
            features: vec!["simd"],
            test_type: TestType::Performance,
            description: "SIMD vector operations",
        },
        TestSuite {
            name: "HNSW Performance Tests",
            package: Some("cheungfun-integrations"),
            features: vec!["hnsw"],
            test_type: TestType::Performance,
            description: "Approximate nearest neighbor search",
        },
        TestSuite {
            name: "Storage Performance Tests",
            package: Some("cheungfun-integrations"),
            features: vec!["storage"],
            test_type: TestType::Performance,
            description: "Database operations",
        },
    ]
}

/// Get storage-specific test suites.
#[must_use]
pub fn get_storage_test_suites() -> Vec<TestSuite> {
    vec![TestSuite {
        name: "Storage Integration Tests",
        package: Some("cheungfun-integrations"),
        features: vec!["storage"],
        test_type: TestType::All,
        description: "Complete storage system testing",
    }]
}

/// Get memory-specific test suites.
#[must_use]
pub fn get_memory_test_suites() -> Vec<TestSuite> {
    vec![TestSuite {
        name: "Memory Integration Tests",
        package: Some("cheungfun-query"),
        features: vec![],
        test_type: TestType::All,
        description: "Complete memory system testing",
    }]
}

/// Get config-specific test suites.
#[must_use]
pub fn get_config_test_suites() -> Vec<TestSuite> {
    vec![TestSuite {
        name: "Config Integration Tests",
        package: Some("cheungfun-core"),
        features: vec![],
        test_type: TestType::All,
        description: "Complete configuration system testing",
    }]
}

/// Print help message.
pub fn print_help() {
    println!("Cheungfun Test Runner");
    println!();
    println!("USAGE:");
    println!("    cargo run --package cheungfun-tools --bin run_tests [CATEGORY]");
    println!();
    println!("CATEGORIES:");
    println!("    unit         Run unit tests only");
    println!("    integration  Run integration tests only");
    println!("    performance  Run performance tests only");
    println!("    storage      Run storage-specific tests");
    println!("    memory       Run memory-specific tests");
    println!("    config       Run configuration-specific tests");
    println!("    all          Run all test categories");
    println!("    help         Show this help message");
    println!();
    println!("EXAMPLES:");
    println!("    cargo run --package cheungfun-tools --bin run_tests unit");
    println!("    cargo run --package cheungfun-tools --bin run_tests storage");
    println!("    cargo run --package cheungfun-tools --bin run_tests all");
    println!();
    println!("If no category is specified, runs default test suite (unit + basic integration).");
}
