//! Basic integration demonstration without AST parsing.
//!
//! This example shows the enhanced code loading infrastructure
//! with AST parsing temporarily disabled to test the integration.

use cheungfun_indexing::{
    loaders::{CodeLoader, CodeLoaderConfig},
    prelude::*,
};
use temp_dir::TempDir;
use tokio::fs;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("🔬 Basic Integration Demo");
    println!("=========================");

    // Create temporary directory with sample code
    let temp_dir = TempDir::new()?;
    create_sample_files(&temp_dir).await?;

    // Example 1: Basic code loading with enhanced infrastructure
    println!("\n📊 Example 1: Enhanced Code Loading Infrastructure");
    await_enhanced_loading(&temp_dir).await?;

    // Example 2: Language detection and metadata
    println!("\n🔍 Example 2: Language Detection and Metadata");
    await_metadata_extraction(&temp_dir).await?;

    // Example 3: Configuration options
    println!("\n⚙️ Example 3: Configuration Options");
    await_configuration_demo(&temp_dir).await?;

    // Example 4: AST Parsing Demo
    println!("\n🌳 Example 4: AST Parsing Demo");
    await_ast_parsing_demo(&temp_dir).await?;

    println!("\n✅ Basic integration demo completed!");
    println!("\n💡 This demonstrates the enhanced code loading infrastructure");
    println!("   with working AST integration using tree-sitter 0.25!");

    Ok(())
}

/// Demonstrate enhanced code loading infrastructure.
async fn await_enhanced_loading(temp_dir: &TempDir) -> Result<(), Box<dyn std::error::Error>> {
    let rust_file = temp_dir.path().join("sample.rs");

    // Load with enhanced infrastructure using AST parsing
    let config = CodeLoaderConfig {
        extract_functions: true,
        extract_classes: true,
        extract_imports: true,
        extract_comments: true,
        ..Default::default()
    };

    let loader = CodeLoader::with_config(&rust_file, config)?;
    let docs = loader.load().await?;

    if let Some(doc) = docs.first() {
        println!("   📄 Document loaded successfully:");
        println!("      ID: {}", doc.id);
        println!("      Content length: {} chars", doc.content.len());
        println!(
            "      Content preview: {}...",
            doc.content.chars().take(100).collect::<String>()
        );

        print_metadata(&doc.metadata, "Enhanced Infrastructure");
    }

    Ok(())
}

/// Demonstrate metadata extraction.
async fn await_metadata_extraction(temp_dir: &TempDir) -> Result<(), Box<dyn std::error::Error>> {
    let files = vec![("sample.rs", "Rust"), ("sample.py", "Python")];

    for (filename, _expected_lang) in files {
        let file_path = temp_dir.path().join(filename);
        let config = CodeLoaderConfig {
            extract_comments: true, // Enable comment extraction
            ..Default::default()
        };

        let loader = CodeLoader::with_config(&file_path, config)?;
        let docs = loader.load().await?;

        if let Some(doc) = docs.first() {
            println!("   📄 File: {}", filename);

            if let Some(language) = doc.metadata.get("language") {
                println!("      Language: {}", language);
            }

            if let Some(loc) = doc.metadata.get("lines_of_code") {
                println!("      Lines of code: {}", loc);
            }

            if let Some(total_lines) = doc.metadata.get("total_lines") {
                println!("      Total lines: {}", total_lines);
            }

            // Show basic extracted metadata
            if let Some(functions) = doc.metadata.get("functions") {
                if let Some(func_array) = functions.as_array() {
                    println!(
                        "      Functions: {} found (basic extraction)",
                        func_array.len()
                    );
                }
            }
        }
    }

    Ok(())
}

/// Demonstrate configuration options.
async fn await_configuration_demo(temp_dir: &TempDir) -> Result<(), Box<dyn std::error::Error>> {
    let rust_file = temp_dir.path().join("sample.rs");

    // Test different configurations
    let configs = vec![
        (
            "Minimal",
            CodeLoaderConfig {
                extract_functions: false,
                extract_classes: false,
                extract_imports: false,
                extract_comments: false,
                ..Default::default()
            },
        ),
        (
            "Functions Only",
            CodeLoaderConfig {
                extract_functions: true,
                extract_classes: false,
                extract_imports: false,
                extract_comments: false,
                ..Default::default()
            },
        ),
        (
            "Full Basic",
            CodeLoaderConfig {
                extract_functions: true,
                extract_classes: true,
                extract_imports: true,
                ..Default::default()
            },
        ),
    ];

    for (name, config) in configs {
        println!("   ⚙️ Configuration: {}", name);

        let loader = CodeLoader::with_config(&rust_file, config)?;
        let docs = loader.load().await?;

        if let Some(doc) = docs.first() {
            let metadata_keys: Vec<_> = doc.metadata.keys().collect();
            println!("      Metadata keys: {:?}", metadata_keys);

            if let Some(functions) = doc.metadata.get("functions") {
                if let Some(func_array) = functions.as_array() {
                    println!("      Functions extracted: {}", func_array.len());
                }
            }
        }
        println!();
    }

    Ok(())
}

/// Print metadata information.
fn print_metadata(metadata: &std::collections::HashMap<String, serde_json::Value>, method: &str) {
    println!("      Method: {}", method);

    if let Some(language) = metadata.get("language") {
        println!("      Language: {}", language);
    }

    if let Some(loc) = metadata.get("lines_of_code") {
        println!("      Lines of code: {}", loc);
    }

    if let Some(total_lines) = metadata.get("total_lines") {
        println!("      Total lines: {}", total_lines);
    }

    if let Some(functions) = metadata.get("functions") {
        if let Some(func_array) = functions.as_array() {
            println!("      Functions: {} found", func_array.len());
            for (i, func) in func_array.iter().take(3).enumerate() {
                if let Some(name) = func.as_str() {
                    println!("        {}. {}", i + 1, name);
                }
            }
        }
    }

    if let Some(classes) = metadata.get("classes") {
        if let Some(class_array) = classes.as_array() {
            println!("      Classes: {} found", class_array.len());
            for (i, class) in class_array.iter().take(3).enumerate() {
                if let Some(name) = class.as_str() {
                    println!("        {}. {}", i + 1, name);
                }
            }
        }
    }
}

/// Create sample code files for demonstration.
async fn create_sample_files(temp_dir: &TempDir) -> Result<(), Box<dyn std::error::Error>> {
    // Create Rust sample
    let rust_content = r#"
use std::collections::HashMap;

/// A simple user struct.
pub struct User {
    pub id: u64,
    pub name: String,
}

impl User {
    /// Create a new user.
    pub fn new(id: u64, name: String) -> Self {
        Self { id, name }
    }
    
    /// Get the user's name.
    pub fn get_name(&self) -> &str {
        &self.name
    }
}

/// User repository.
pub struct UserRepository {
    users: HashMap<u64, User>,
}

impl UserRepository {
    /// Create a new repository.
    pub fn new() -> Self {
        Self {
            users: HashMap::new(),
        }
    }
    
    /// Add a user.
    pub fn add_user(&mut self, user: User) {
        self.users.insert(user.id, user);
    }
}
"#;

    // Create Python sample
    let python_content = r#"
"""Simple user management module."""

class User:
    """A simple user class."""
    
    def __init__(self, user_id: int, name: str):
        """Initialize a user."""
        self.id = user_id
        self.name = name
    
    def get_name(self) -> str:
        """Get the user's name."""
        return self.name

class UserRepository:
    """Repository for managing users."""
    
    def __init__(self):
        """Initialize the repository."""
        self.users = {}
    
    def add_user(self, user: User) -> None:
        """Add a user to the repository."""
        self.users[user.id] = user
    
    def find_user(self, user_id: int) -> User:
        """Find a user by ID."""
        return self.users.get(user_id)

def create_user(user_id: int, name: str) -> User:
    """Create a new user."""
    return User(user_id, name)
"#;

    // Create JavaScript sample
    let js_content = r#"
/**
 * Simple user management module.
 */

class User {
    /**
     * Create a new user.
     * @param {number} id - User ID
     * @param {string} name - User name
     */
    constructor(id, name) {
        this.id = id;
        this.name = name;
    }

    /**
     * Get the user's name.
     * @returns {string} The user's name
     */
    getName() {
        return this.name;
    }
}

class UserRepository {
    /**
     * Create a new repository.
     */
    constructor() {
        this.users = new Map();
    }

    /**
     * Add a user to the repository.
     * @param {User} user - The user to add
     */
    addUser(user) {
        this.users.set(user.id, user);
    }

    /**
     * Find a user by ID.
     * @param {number} id - User ID
     * @returns {User|undefined} The user or undefined
     */
    findUser(id) {
        return this.users.get(id);
    }
}

/**
 * Create a new user.
 * @param {number} id - User ID
 * @param {string} name - User name
 * @returns {User} New user instance
 */
function createUser(id, name) {
    return new User(id, name);
}
"#;

    // Create C# sample
    let cs_content = r#"
using System;
using System.Collections.Generic;

namespace UserManagement
{
    /// <summary>
    /// A simple user class.
    /// </summary>
    public class User
    {
        /// <summary>
        /// Gets or sets the user ID.
        /// </summary>
        public int Id { get; set; }

        /// <summary>
        /// Gets or sets the user name.
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Initializes a new instance of the User class.
        /// </summary>
        /// <param name="id">The user ID.</param>
        /// <param name="name">The user name.</param>
        public User(int id, string name)
        {
            Id = id;
            Name = name;
        }

        /// <summary>
        /// Gets the user's name.
        /// </summary>
        /// <returns>The user's name.</returns>
        public string GetName()
        {
            return Name;
        }
    }

    /// <summary>
    /// Repository for managing users.
    /// </summary>
    public class UserRepository
    {
        private readonly Dictionary<int, User> users;

        /// <summary>
        /// Initializes a new instance of the UserRepository class.
        /// </summary>
        public UserRepository()
        {
            users = new Dictionary<int, User>();
        }

        /// <summary>
        /// Adds a user to the repository.
        /// </summary>
        /// <param name="user">The user to add.</param>
        public void AddUser(User user)
        {
            users[user.Id] = user;
        }

        /// <summary>
        /// Finds a user by ID.
        /// </summary>
        /// <param name="id">The user ID.</param>
        /// <returns>The user or null if not found.</returns>
        public User FindUser(int id)
        {
            users.TryGetValue(id, out User user);
            return user;
        }
    }
}
"#;

    // Create Java sample
    let java_content = r#"
package com.example.usermanagement;

import java.util.HashMap;
import java.util.Map;

/**
 * A simple user class.
 */
public class User {
    private int id;
    private String name;

    /**
     * Creates a new user.
     * @param id The user ID
     * @param name The user name
     */
    public User(int id, String name) {
        this.id = id;
        this.name = name;
    }

    /**
     * Gets the user ID.
     * @return The user ID
     */
    public int getId() {
        return id;
    }

    /**
     * Gets the user name.
     * @return The user name
     */
    public String getName() {
        return name;
    }

    /**
     * Sets the user name.
     * @param name The new name
     */
    public void setName(String name) {
        this.name = name;
    }
}

/**
 * Repository for managing users.
 */
class UserRepository {
    private Map<Integer, User> users;

    /**
     * Creates a new repository.
     */
    public UserRepository() {
        this.users = new HashMap<>();
    }

    /**
     * Adds a user to the repository.
     * @param user The user to add
     */
    public void addUser(User user) {
        users.put(user.getId(), user);
    }

    /**
     * Finds a user by ID.
     * @param id The user ID
     * @return The user or null if not found
     */
    public User findUser(int id) {
        return users.get(id);
    }
}
"#;

    // Create Go sample
    let go_content = r#"
package main

import "fmt"

// User represents a simple user.
type User struct {
    ID   int    `json:"id"`
    Name string `json:"name"`
}

// NewUser creates a new user.
func NewUser(id int, name string) *User {
    return &User{
        ID:   id,
        Name: name,
    }
}

// GetName returns the user's name.
func (u *User) GetName() string {
    return u.Name
}

// SetName sets the user's name.
func (u *User) SetName(name string) {
    u.Name = name
}

// UserRepository manages users.
type UserRepository struct {
    users map[int]*User
}

// NewUserRepository creates a new repository.
func NewUserRepository() *UserRepository {
    return &UserRepository{
        users: make(map[int]*User),
    }
}

// AddUser adds a user to the repository.
func (r *UserRepository) AddUser(user *User) {
    r.users[user.ID] = user
}

// FindUser finds a user by ID.
func (r *UserRepository) FindUser(id int) (*User, bool) {
    user, exists := r.users[id]
    return user, exists
}

// ListUsers returns all users.
func (r *UserRepository) ListUsers() []*User {
    users := make([]*User, 0, len(r.users))
    for _, user := range r.users {
        users = append(users, user)
    }
    return users
}

func main() {
    repo := NewUserRepository()
    user := NewUser(1, "John Doe")
    repo.AddUser(user)

    if foundUser, exists := repo.FindUser(1); exists {
        fmt.Printf("Found user: %s\n", foundUser.GetName())
    }
}
"#;

    // Write all files
    fs::write(temp_dir.path().join("sample.rs"), rust_content).await?;
    fs::write(temp_dir.path().join("sample.py"), python_content).await?;
    fs::write(temp_dir.path().join("sample.js"), js_content).await?;
    fs::write(temp_dir.path().join("sample.cs"), cs_content).await?;
    fs::write(temp_dir.path().join("sample.java"), java_content).await?;
    fs::write(temp_dir.path().join("sample.go"), go_content).await?;

    Ok(())
}

/// Demonstrate AST parsing functionality.
async fn await_ast_parsing_demo(temp_dir: &TempDir) -> Result<(), Box<dyn std::error::Error>> {
    let files = [
        ("sample.rs", "Rust"),
        ("sample.py", "Python"),
        ("sample.js", "JavaScript"),
        ("sample.cs", "C#"),
        ("sample.java", "Java"),
        ("sample.go", "Go"),
    ];

    for (filename, language_name) in files {
        let file_path = temp_dir.path().join(filename);

        // Test with full AST parsing enabled
        let config = CodeLoaderConfig {
            extract_functions: true,
            extract_classes: true,
            extract_imports: true,
            extract_comments: true,
            ..Default::default()
        };

        let loader = CodeLoader::with_config(&file_path, config)?;

        match loader.load().await {
            Ok(documents) => {
                if let Some(doc) = documents.first() {
                    println!("   📄 File: {}", filename);
                    println!("      Language: {:?}", doc.metadata.get("language"));
                    println!("      Total lines: {:?}", doc.metadata.get("total_lines"));

                    // Show AST-extracted metadata
                    if let Some(functions) = doc.metadata.get("functions") {
                        if let Some(func_array) = functions.as_array() {
                            println!(
                                "      Functions: {} found (AST extraction)",
                                func_array.len()
                            );
                            for (i, func) in func_array.iter().take(3).enumerate() {
                                if let Some(name) = func.as_str() {
                                    println!("        {}. {}", i + 1, name);
                                }
                            }
                        }
                    }

                    if let Some(classes) = doc.metadata.get("classes") {
                        if let Some(class_array) = classes.as_array() {
                            println!(
                                "      Classes: {} found (AST extraction)",
                                class_array.len()
                            );
                            for (i, class) in class_array.iter().take(3).enumerate() {
                                if let Some(name) = class.as_str() {
                                    println!("        {}. {}", i + 1, name);
                                }
                            }
                        }
                    }

                    if let Some(imports) = doc.metadata.get("imports") {
                        if let Some(import_array) = imports.as_array() {
                            println!(
                                "      Imports: {} found (AST extraction)",
                                import_array.len()
                            );
                        }
                    }

                    if let Some(comments) = doc.metadata.get("comments") {
                        if let Some(comment_array) = comments.as_array() {
                            println!(
                                "      Comments: {} found (AST extraction)",
                                comment_array.len()
                            );
                        }
                    }
                }
            }
            Err(e) => {
                println!("   ❌ Failed to parse {} with AST: {}", filename, e);
                println!("      This is expected if tree-sitter queries need refinement");
            }
        }
    }

    println!();
    Ok(())
}
