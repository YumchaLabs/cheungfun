//! SQLx-based storage implementations.
//!
//! This module provides database storage implementations using SQLx,
//! supporting PostgreSQL and SQLite backends with connection pooling
//! and automatic migrations.

pub mod chat_store;
pub mod document_store;
pub mod index_store;
pub mod migrations;

pub use chat_store::SqlxChatStore;
pub use document_store::SqlxDocumentStore;
pub use index_store::SqlxIndexStore;

use sqlx::{Pool, Postgres, Sqlite};
use std::fmt;

/// Database connection pool type.
#[derive(Debug, Clone)]
pub enum DatabasePool {
    /// PostgreSQL connection pool.
    Postgres(Pool<Postgres>),
    /// SQLite connection pool.
    Sqlite(Pool<Sqlite>),
}

impl DatabasePool {
    /// Create a new PostgreSQL pool from connection string.
    pub async fn postgres(database_url: &str) -> Result<Self, sqlx::Error> {
        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(10)
            .connect(database_url)
            .await?;
        Ok(Self::Postgres(pool))
    }

    /// Create a new SQLite pool from connection string.
    pub async fn sqlite(database_url: &str) -> Result<Self, sqlx::Error> {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(5)
            .connect(database_url)
            .await?;
        Ok(Self::Sqlite(pool))
    }

    /// Run database migrations.
    pub async fn migrate(&self) -> Result<(), sqlx::Error> {
        match self {
            Self::Postgres(pool) => {
                sqlx::migrate!("./migrations/postgres").run(pool).await?;
            }
            Self::Sqlite(pool) => {
                sqlx::migrate!("./migrations/sqlite").run(pool).await?;
            }
        }
        Ok(())
    }

    /// Check if the database connection is healthy.
    pub async fn health_check(&self) -> Result<(), sqlx::Error> {
        match self {
            Self::Postgres(pool) => {
                sqlx::query("SELECT 1").execute(pool).await?;
            }
            Self::Sqlite(pool) => {
                sqlx::query("SELECT 1").execute(pool).await?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for DatabasePool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Postgres(_) => write!(f, "PostgreSQL"),
            Self::Sqlite(_) => write!(f, "SQLite"),
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
    /// Whether to run migrations automatically.
    pub auto_migrate: bool,
    /// Maximum number of connections in the pool.
    pub max_connections: u32,
}

impl Default for SqlxStorageConfig {
    fn default() -> Self {
        Self {
            database_url: "sqlite::memory:".to_string(),
            table_prefix: "cheungfun_".to_string(),
            auto_migrate: true,
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

    /// Set whether to run migrations automatically.
    pub fn with_auto_migrate(mut self, auto_migrate: bool) -> Self {
        self.auto_migrate = auto_migrate;
        self
    }

    /// Set the maximum number of connections.
    pub fn with_max_connections(mut self, max_connections: u32) -> Self {
        self.max_connections = max_connections;
        self
    }

    /// Create a database pool from this configuration.
    pub async fn create_pool(&self) -> Result<DatabasePool, sqlx::Error> {
        if self.database_url.starts_with("postgres://")
            || self.database_url.starts_with("postgresql://")
        {
            let pool = sqlx::postgres::PgPoolOptions::new()
                .max_connections(self.max_connections)
                .connect(&self.database_url)
                .await?;
            Ok(DatabasePool::Postgres(pool))
        } else {
            let pool = sqlx::sqlite::SqlitePoolOptions::new()
                .max_connections(self.max_connections)
                .connect(&self.database_url)
                .await?;
            Ok(DatabasePool::Sqlite(pool))
        }
    }
}
