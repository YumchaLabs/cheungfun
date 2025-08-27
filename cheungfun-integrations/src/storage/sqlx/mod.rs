//! SQLx-based storage implementations.
//!
//! This module provides database connection utilities and re-exports
//! the SQLx-based KVStore implementation. The old direct SQLx implementations
//! have been replaced with KVStore-based implementations.

// Re-export the SQLx KVStore implementation and DatabasePool
pub use crate::storage::kvstore::sqlx::{SqlxKVStore, DatabasePool};

// Helper functions for creating database pools and configurations.

/// Helper functions for creating database pools and configurations.
pub struct SqlxHelper;

impl SqlxHelper {
    /// Create a new PostgreSQL pool from connection string.
    pub async fn postgres_pool(database_url: &str) -> Result<DatabasePool, sqlx::Error> {
        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(10)
            .connect(database_url)
            .await?;
        Ok(DatabasePool::Postgres(pool))
    }

    /// Create a new SQLite pool from connection string.
    pub async fn sqlite_pool(database_url: &str) -> Result<DatabasePool, sqlx::Error> {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(5)
            .connect(database_url)
            .await?;
        Ok(DatabasePool::Sqlite(pool))
    }

    /// Create a database pool from URL (auto-detects type).
    pub async fn create_pool(database_url: &str) -> Result<DatabasePool, sqlx::Error> {
        if database_url.starts_with("postgres://") || database_url.starts_with("postgresql://") {
            Self::postgres_pool(database_url).await
        } else {
            Self::sqlite_pool(database_url).await
        }
    }
}

/// Configuration for SQLx storage.
#[derive(Debug, Clone)]
pub struct SqlxStorageConfig {
    /// Database connection URL.
    pub database_url: String,
    /// Table name prefix.
    pub table_prefix: String,
    /// Maximum number of connections in the pool.
    pub max_connections: u32,
}

impl Default for SqlxStorageConfig {
    fn default() -> Self {
        Self {
            database_url: "sqlite::memory:".to_string(),
            table_prefix: "cheungfun_".to_string(),
            max_connections: 10,
        }
    }
}

impl SqlxStorageConfig {
    /// Create a new configuration with the given database URL.
    pub fn new<S: Into<String>>(database_url: S) -> Self {
        Self {
            database_url: database_url.into(),
            ..Default::default()
        }
    }

    /// Set the table prefix.
    pub fn with_table_prefix<S: Into<String>>(mut self, prefix: S) -> Self {
        self.table_prefix = prefix.into();
        self
    }

    /// Set the maximum number of connections.
    pub fn with_max_connections(mut self, max_connections: u32) -> Self {
        self.max_connections = max_connections;
        self
    }

    /// Create a SqlxKVStore from this configuration.
    pub async fn create_kv_store(&self) -> Result<SqlxKVStore, sqlx::Error> {
        let pool = SqlxHelper::create_pool(&self.database_url).await?;
        SqlxKVStore::new(pool, &self.table_prefix).await.map_err(|e| {
            sqlx::Error::Configuration(format!("Failed to create KV store: {}", e).into())
        })
    }
}
