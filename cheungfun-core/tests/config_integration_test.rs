//! Integration tests for the configuration system.

use cheungfun_core::config::{ConfigManager, EmbedderConfig, LlmConfig};
use tempfile::TempDir;
use tokio::fs;

#[tokio::test]
async fn test_config_manager_basic_operations() {
    let temp_dir = TempDir::new().unwrap();
    let config_dir = temp_dir.path();

    // Create test configuration files
    let database_config = serde_json::json!({
        "url": "sqlite://test.db",
        "max_connections": 10,
        "auto_migrate": true
    });

    let embedding_config = serde_json::json!({
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "batch_size": 32
    });

    // Write config files
    fs::write(
        config_dir.join("database.json"),
        serde_json::to_string_pretty(&database_config).unwrap(),
    )
    .await
    .unwrap();

    fs::write(
        config_dir.join("embedding.json"),
        serde_json::to_string_pretty(&embedding_config).unwrap(),
    )
    .await
    .unwrap();

    // Load configuration
    let mut config_manager = ConfigManager::new();
    config_manager
        .load_from_directory(config_dir)
        .await
        .unwrap();

    // Test getting configuration values
    let db_url = config_manager.get_string("database.url").unwrap();
    assert_eq!(db_url, "sqlite://test.db");

    let max_connections = config_manager.get_u32("database.max_connections").unwrap();
    assert_eq!(max_connections, 10);

    let auto_migrate = config_manager.get_bool("database.auto_migrate").unwrap();
    assert!(auto_migrate);

    let embedding_model = config_manager.get_string("embedding.model").unwrap();
    assert_eq!(embedding_model, "sentence-transformers/all-MiniLM-L6-v2");

    let dimension = config_manager.get_u32("embedding.dimension").unwrap();
    assert_eq!(dimension, 384);
}

#[tokio::test]
async fn test_config_manager_environment_variables() {
    let temp_dir = TempDir::new().unwrap();
    let config_dir = temp_dir.path();

    // Create config with environment variable substitution
    let config_with_env = serde_json::json!({
        "database_url": "${TEST_DB_URL:sqlite://default.db}",
        "api_key": "${TEST_API_KEY:default_key}",
        "port": "${TEST_PORT:8080}"
    });

    fs::write(
        config_dir.join("app.json"),
        serde_json::to_string_pretty(&config_with_env).unwrap(),
    )
    .await
    .unwrap();

    // Set environment variable
    unsafe {
        std::env::set_var("TEST_DB_URL", "postgresql://localhost/test");
    }

    let mut config_manager = ConfigManager::new();
    config_manager
        .load_from_directory(config_dir)
        .await
        .unwrap();

    // Test environment variable substitution
    let db_url = config_manager.get_string("app.database_url").unwrap();
    assert_eq!(db_url, "postgresql://localhost/test");

    // Test default value when env var is not set
    let api_key = config_manager.get_string("app.api_key").unwrap();
    assert_eq!(api_key, "default_key");

    // Clean up
    unsafe {
        std::env::remove_var("TEST_DB_URL");
    }
}

#[tokio::test]
async fn test_config_manager_environment_overrides() {
    let temp_dir = TempDir::new().unwrap();
    let config_dir = temp_dir.path();

    let config = serde_json::json!({
        "setting1": "original_value",
        "setting2": 42
    });

    fs::write(
        config_dir.join("test.json"),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .await
    .unwrap();

    let mut config_manager = ConfigManager::new();
    config_manager
        .load_from_directory(config_dir)
        .await
        .unwrap();

    // Test original values
    assert_eq!(
        config_manager.get_string("test.setting1").unwrap(),
        "original_value"
    );
    assert_eq!(config_manager.get_u32("test.setting2").unwrap(), 42);

    // Set environment overrides
    config_manager.set_env_override("test.setting1", "overridden_value");
    config_manager.set_env_override("test.setting2", "100");

    // Test overridden values
    assert_eq!(
        config_manager.get_string("test.setting1").unwrap(),
        "overridden_value"
    );
    assert_eq!(config_manager.get_string("test.setting2").unwrap(), "100");

    // Remove override
    config_manager.remove_env_override("test.setting1");
    assert_eq!(
        config_manager.get_string("test.setting1").unwrap(),
        "original_value"
    );
}

#[tokio::test]
async fn test_config_manager_namespace_operations() {
    let temp_dir = TempDir::new().unwrap();
    let config_dir = temp_dir.path();

    let config = serde_json::json!({
        "level1": {
            "level2": {
                "setting": "deep_value"
            },
            "array": [1, 2, 3],
            "boolean": true
        },
        "top_level": "surface_value"
    });

    fs::write(
        config_dir.join("nested.json"),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .await
    .unwrap();

    let mut config_manager = ConfigManager::new();
    config_manager
        .load_from_directory(config_dir)
        .await
        .unwrap();

    // Test nested value access
    let deep_value = config_manager
        .get_string("nested.level1.level2.setting")
        .unwrap();
    assert_eq!(deep_value, "deep_value");

    let top_value = config_manager.get_string("nested.top_level").unwrap();
    assert_eq!(top_value, "surface_value");

    let boolean_value = config_manager.get_bool("nested.level1.boolean").unwrap();
    assert!(boolean_value);

    // Test namespace keys
    let keys = config_manager.get_namespace_keys("nested").unwrap();
    assert!(keys.contains(&"nested.level1".to_string()));
    assert!(keys.contains(&"nested.level1.level2".to_string()));
    assert!(keys.contains(&"nested.level1.level2.setting".to_string()));
    assert!(keys.contains(&"nested.top_level".to_string()));
}

#[tokio::test]
async fn test_config_manager_statistics() {
    let temp_dir = TempDir::new().unwrap();
    let config_dir = temp_dir.path();

    // Create multiple config files
    for i in 0..3 {
        let config = serde_json::json!({
            "setting": format!("value_{}", i)
        });

        fs::write(
            config_dir.join(format!("config_{}.json", i)),
            serde_json::to_string_pretty(&config).unwrap(),
        )
        .await
        .unwrap();
    }

    let mut config_manager = ConfigManager::new();
    config_manager
        .load_from_directory(config_dir)
        .await
        .unwrap();

    // Set some environment overrides
    config_manager.set_env_override("test.key1", "value1");
    config_manager.set_env_override("test.key2", "value2");

    let stats = config_manager.get_stats();
    assert_eq!(stats.total_namespaces, 3);
    assert_eq!(stats.total_files, 3);
    assert_eq!(stats.env_overrides, 2);
    assert!(!stats.hot_reload_enabled); // Not enabled in this test
}

#[tokio::test]
async fn test_json_configurable_embedder_config() {
    let temp_dir = TempDir::new().unwrap();
    let config_file = temp_dir.path().join("embedder.json");

    // Create a Candle embedder config
    let embedder_config = EmbedderConfig::candle("sentence-transformers/all-MiniLM-L6-v2", "cpu");

    // Test JSON serialization
    let json_str = serde_json::to_string_pretty(&embedder_config).unwrap();
    tokio::fs::write(&config_file, json_str).await.unwrap();

    // Test loading from JSON
    let json_content = tokio::fs::read_to_string(&config_file).await.unwrap();
    let loaded_config: EmbedderConfig = serde_json::from_str(&json_content).unwrap();

    // Test validation
    loaded_config.validate().unwrap();

    // Test that the loaded config matches the original
    match (&embedder_config, &loaded_config) {
        (
            EmbedderConfig::Candle { model_name, .. },
            EmbedderConfig::Candle {
                model_name: loaded_model,
                ..
            },
        ) => {
            assert_eq!(model_name, loaded_model);
        }
        _ => panic!("Config types don't match"),
    }
}

#[tokio::test]
async fn test_json_configurable_llm_config() {
    let temp_dir = TempDir::new().unwrap();
    let config_file = temp_dir.path().join("llm.json");

    // Create an OpenAI LLM config
    let llm_config = LlmConfig::openai("gpt-3.5-turbo", "test-key")
        .with_temperature(0.7)
        .with_max_tokens(1000);

    // Test JSON serialization
    let json_str = serde_json::to_string_pretty(&llm_config).unwrap();
    tokio::fs::write(&config_file, json_str).await.unwrap();

    // Test loading from JSON
    let json_content = tokio::fs::read_to_string(&config_file).await.unwrap();
    let loaded_config: LlmConfig = serde_json::from_str(&json_content).unwrap();

    assert_eq!(loaded_config.model, llm_config.model);
    assert_eq!(loaded_config.api_key, llm_config.api_key);
    assert_eq!(loaded_config.temperature, llm_config.temperature);

    // Test validation
    loaded_config.validate().unwrap();

    // Test invalid temperature
    let mut invalid_config = LlmConfig::openai("test-model", "test-key");
    invalid_config.temperature = Some(3.0); // Invalid: temperature > 2.0

    assert!(invalid_config.validate().is_err());
}

#[tokio::test]
async fn test_config_serialization() {
    // Test that different config types can be serialized and deserialized
    let configs = vec![
        EmbedderConfig::candle("sentence-transformers/all-MiniLM-L6-v2", "cpu"),
        EmbedderConfig::api("openai", "text-embedding-ada-002", "test-key"),
        EmbedderConfig::custom("custom-implementation"),
    ];

    for config in configs {
        // Test serialization
        let json_str = serde_json::to_string_pretty(&config).unwrap();

        // Test deserialization
        let loaded_config: EmbedderConfig = serde_json::from_str(&json_str).unwrap();

        // Test validation
        loaded_config.validate().unwrap();

        // Test that the configs are equivalent
        let original_json = serde_json::to_string(&config).unwrap();
        let loaded_json = serde_json::to_string(&loaded_config).unwrap();
        assert_eq!(original_json, loaded_json);
    }
}

#[tokio::test]
async fn test_config_error_handling() {
    let config_manager = ConfigManager::new();

    // Test getting non-existent key
    let result = config_manager.get_string("nonexistent.key");
    assert!(result.is_err());

    // Test getting wrong type
    let temp_dir = TempDir::new().unwrap();
    let config_dir = temp_dir.path();

    let config = serde_json::json!({
        "string_value": "hello",
        "number_value": 42
    });

    fs::write(
        config_dir.join("test.json"),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .await
    .unwrap();

    let mut config_manager = ConfigManager::new();
    config_manager
        .load_from_directory(config_dir)
        .await
        .unwrap();

    // Try to get string as number
    let result = config_manager.get_u32("test.string_value");
    assert!(result.is_err());

    // Try to get number as boolean
    let result = config_manager.get_bool("test.number_value");
    assert!(result.is_err());
}
