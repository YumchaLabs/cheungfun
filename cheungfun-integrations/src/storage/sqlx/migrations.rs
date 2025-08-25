//! Database migrations for SQLx storage.

use cheungfun_core::{CheungfunError, Result};
use sqlx::{Pool, Postgres, Row, Sqlite};
use tracing::{debug, info};

use super::DatabasePool;

/// Migration manager for SQLx storage.
pub struct MigrationManager {
    pool: DatabasePool,
}

impl MigrationManager {
    /// Create a new migration manager.
    pub fn new(pool: DatabasePool) -> Self {
        Self { pool }
    }

    /// Run all migrations.
    pub async fn migrate(&self) -> Result<()> {
        match &self.pool {
            DatabasePool::Postgres(pool) => {
                self.migrate_postgres(pool).await?;
            }
            DatabasePool::Sqlite(pool) => {
                self.migrate_sqlite(pool).await?;
            }
        }
        info!("Database migrations completed successfully");
        Ok(())
    }

    /// Run PostgreSQL migrations.
    async fn migrate_postgres(&self, pool: &Pool<Postgres>) -> Result<()> {
        debug!("Running PostgreSQL migrations");

        // Create documents table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS cheungfun_documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata JSONB,
                content_hash TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
            "#,
        )
        .execute(pool)
        .await
        .map_err(|e| CheungfunError::VectorStore {
            message: format!("Failed to create documents table: {}", e),
        })?;

        // Create conversations table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS cheungfun_conversations (
                id SERIAL PRIMARY KEY,
                conversation_key TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB,
                message_order INTEGER NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_cheungfun_conversations_key 
                ON cheungfun_conversations (conversation_key);
            CREATE INDEX IF NOT EXISTS idx_cheungfun_conversations_order 
                ON cheungfun_conversations (conversation_key, message_order);
            "#,
        )
        .execute(pool)
        .await
        .map_err(|e| CheungfunError::VectorStore {
            message: format!("Failed to create conversations table: {}", e),
        })?;

        // Create indexes table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS cheungfun_indexes (
                index_id TEXT PRIMARY KEY,
                index_type TEXT NOT NULL,
                config JSONB NOT NULL,
                node_ids JSONB NOT NULL,
                metadata JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            )
            "#,
        )
        .execute(pool)
        .await
        .map_err(|e| CheungfunError::VectorStore {
            message: format!("Failed to create indexes table: {}", e),
        })?;

        debug!("PostgreSQL migrations completed");
        Ok(())
    }

    /// Run SQLite migrations.
    async fn migrate_sqlite(&self, pool: &Pool<Sqlite>) -> Result<()> {
        debug!("Running SQLite migrations");

        // Create documents table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS cheungfun_documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT,
                content_hash TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            "#,
        )
        .execute(pool)
        .await
        .map_err(|e| CheungfunError::VectorStore {
            message: format!("Failed to create documents table: {}", e),
        })?;

        // Create conversations table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS cheungfun_conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_key TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                message_order INTEGER NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_cheungfun_conversations_key 
                ON cheungfun_conversations (conversation_key);
            CREATE INDEX IF NOT EXISTS idx_cheungfun_conversations_order 
                ON cheungfun_conversations (conversation_key, message_order);
            "#,
        )
        .execute(pool)
        .await
        .map_err(|e| CheungfunError::VectorStore {
            message: format!("Failed to create conversations table: {}", e),
        })?;

        // Create indexes table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS cheungfun_indexes (
                index_id TEXT PRIMARY KEY,
                index_type TEXT NOT NULL,
                config TEXT NOT NULL,
                node_ids TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
            "#,
        )
        .execute(pool)
        .await
        .map_err(|e| CheungfunError::VectorStore {
            message: format!("Failed to create indexes table: {}", e),
        })?;

        debug!("SQLite migrations completed");
        Ok(())
    }

    /// Check if migrations are needed.
    pub async fn needs_migration(&self) -> Result<bool> {
        match &self.pool {
            DatabasePool::Postgres(pool) => {
                let result = sqlx::query(
                    r#"
                    SELECT COUNT(*) as count FROM information_schema.tables 
                    WHERE table_name IN ('cheungfun_documents', 'cheungfun_conversations', 'cheungfun_indexes')
                    "#,
                )
                .fetch_one(pool)
                .await
                .map_err(|e| CheungfunError::VectorStore {
                    message: format!("Failed to check migration status: {}", e),
                })?;

                let count: i64 = result.get("count");
                Ok(count < 3)
            }
            DatabasePool::Sqlite(pool) => {
                let result = sqlx::query(
                    r#"
                    SELECT COUNT(*) as count FROM sqlite_master 
                    WHERE type='table' AND name IN ('cheungfun_documents', 'cheungfun_conversations', 'cheungfun_indexes')
                    "#,
                )
                .fetch_one(pool)
                .await
                .map_err(|e| CheungfunError::VectorStore {
                    message: format!("Failed to check migration status: {}", e),
                })?;

                let count: i64 = result.get("count");
                Ok(count < 3)
            }
        }
    }

    /// Get migration status.
    pub async fn migration_status(&self) -> Result<MigrationStatus> {
        let needs_migration = self.needs_migration().await?;

        if needs_migration {
            Ok(MigrationStatus::Pending)
        } else {
            Ok(MigrationStatus::UpToDate)
        }
    }
}

/// Migration status.
#[derive(Debug, Clone, PartialEq)]
pub enum MigrationStatus {
    /// Migrations are up to date.
    UpToDate,
    /// Migrations are pending.
    Pending,
}

impl std::fmt::Display for MigrationStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UpToDate => write!(f, "Up to date"),
            Self::Pending => write!(f, "Pending"),
        }
    }
}
