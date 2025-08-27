//! Code Parsing with AST Demo
//!
//! Demonstrates the AST-based code parsing capabilities using tree-sitter.
//! Shows how to extract functions, classes, imports, and other code structures.

use cheungfun_indexing::{
    loaders::ProgrammingLanguage,
    parsers::{AstAnalysis, AstParser, AstParserConfig},
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌳 AST Code Parsing Demo");
    println!("=========================\n");

    // Sample Rust code for analysis
    let rust_code = r#"
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// User management system
#[derive(Debug, Serialize, Deserialize)]
pub struct User {
    pub id: u64,
    pub name: String,
    pub email: String,
    pub roles: Vec<String>,
}

impl User {
    /// Creates a new user with the given details
    pub fn new(id: u64, name: String, email: String) -> Self {
        Self {
            id,
            name,
            email,
            roles: Vec::new(),
        }
    }

    /// Adds a role to the user
    pub fn add_role(&mut self, role: String) {
        if !self.roles.contains(&role) {
            self.roles.push(role);
        }
    }

    /// Checks if user has a specific role
    pub fn has_role(&self, role: &str) -> bool {
        self.roles.contains(&role.to_string())
    }
}

/// User repository for database operations
pub struct UserRepository {
    users: HashMap<u64, User>,
}

impl UserRepository {
    pub fn new() -> Self {
        Self {
            users: HashMap::new(),
        }
    }

    pub fn add_user(&mut self, user: User) -> Result<(), String> {
        if self.users.contains_key(&user.id) {
            return Err("User already exists".to_string());
        }
        self.users.insert(user.id, user);
        Ok(())
    }
}
"#;

    // Sample C# code for analysis
    let csharp_code = r#"
using System;
using System.Collections.Generic;
using UnityEngine;

namespace GameSystem
{
    /// <summary>
    /// Player controller for handling movement and actions
    /// </summary>
    public class PlayerController : MonoBehaviour
    {
        [SerializeField] private float moveSpeed = 5.0f;
        [SerializeField] private float jumpForce = 10.0f;
        
        private Rigidbody rb;
        private bool isGrounded;

        private void Start()
        {
            rb = GetComponent<Rigidbody>();
        }

        private void Update()
        {
            HandleMovement();
            HandleJump();
        }

        /// <summary>
        /// Handles player movement input
        /// </summary>
        private void HandleMovement()
        {
            float horizontal = Input.GetAxis("Horizontal");
            float vertical = Input.GetAxis("Vertical");
            
            Vector3 direction = new Vector3(horizontal, 0, vertical);
            transform.Translate(direction * moveSpeed * Time.deltaTime);
        }

        /// <summary>
        /// Handles jump input and physics
        /// </summary>
        private void HandleJump()
        {
            if (Input.GetButtonDown("Jump") && isGrounded)
            {
                rb.AddForce(Vector3.up * jumpForce, ForceMode.Impulse);
                isGrounded = false;
            }
        }
    }
}
"#;

    // Test Rust parsing
    println!("🦀 Analyzing Rust Code");
    println!("-".repeat(40));
    analyze_code(rust_code, ProgrammingLanguage::Rust).await?;
    println!();

    // Test C# parsing
    println!("🎮 Analyzing C# Code");
    println!("-".repeat(40));
    analyze_code(csharp_code, ProgrammingLanguage::CSharp).await?;

    Ok(())
}

async fn analyze_code(
    code: &str,
    language: ProgrammingLanguage,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = AstParserConfig {
        extract_functions: true,
        extract_classes: true,
        extract_imports: true,
        extract_comments: true,
        extract_variables: true,
        ..Default::default()
    };

    let mut parser = AstParser::new(language, config)?;
    let analysis = parser.parse_content(code, "example_file").await?;

    println!("📊 Analysis Results:");
    println!("  Lines of code: {}", code.lines().count());
    println!("  Characters: {}", code.len());

    if !analysis.imports.is_empty() {
        println!("\n📦 Imports/Uses:");
        for import in &analysis.imports {
            println!("  - {}", import);
        }
    }

    if !analysis.classes.is_empty() {
        println!("\n🏗️  Classes/Structs:");
        for class in &analysis.classes {
            println!("  - {} (line {})", class.name, class.line_number);
            if !class.methods.is_empty() {
                println!("    Methods: {}", class.methods.join(", "));
            }
        }
    }

    if !analysis.functions.is_empty() {
        println!("\n⚡ Functions:");
        for function in &analysis.functions {
            println!(
                "  - {} (line {}, {} params)",
                function.name,
                function.line_number,
                function.parameters.len()
            );
        }
    }

    if !analysis.comments.is_empty() {
        println!("\n💬 Comments:");
        for comment in analysis.comments.iter().take(3) {
            let preview = comment.content.lines().next().unwrap_or("");
            println!("  - Line {}: {}", comment.line_number, preview);
        }
        if analysis.comments.len() > 3 {
            println!("  ... and {} more comments", analysis.comments.len() - 3);
        }
    }

    println!("\n📈 Complexity Metrics:");
    println!("  Functions: {}", analysis.functions.len());
    println!("  Classes: {}", analysis.classes.len());
    println!("  Comments: {}", analysis.comments.len());

    Ok(())
}
