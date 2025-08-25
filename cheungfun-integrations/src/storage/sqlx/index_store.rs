//! SQLx-based index store implementation.

use async_trait::async_trait;
use cheungfun_core::{
    traits::{IndexStore, IndexStruct},
    CheungfunError, Result,
};
use sqlx::Row;
use tracing::{debug, info};

use super::{DatabasePool, SqlxStorageConfig};

/// SQLx-based index store implementation.
///
/// This store persists index structures and metadata in a relational database using SQLx,
/// supporting both PostgreSQL and SQLite backends.
#[derive(Debug)]
pub struct SqlxIndexStore {
    pool: DatabasePool,
    table_name: String,
}

impl SqlxIndexStore {
    /// Create a new SQLx index store.
    pub async fn new(config: SqlxStorageConfig) -> Result<Self> {
        let pool = config
            .create_pool()
            .await
            .map_err(|e| CheungfunError::Configuration {
                message: format!("Failed to create database pool: {}", e),
            })?;

        let table_name = format!("{}indexes", config.table_prefix);

        let store = Self {
            pool,
            table_name: table_name.clone(),
        };

        if config.auto_migrate {
            store.create_table().await?;
        }

        info!("Created SQLx index store with table: {}", table_name);
        Ok(store)
    }

    /// Create a new index store with an existing pool.
    pub fn with_pool(pool: DatabasePool, table_prefix: &str) -> Self {
        let table_name = format!("{}indexes", table_prefix);
        Self { pool, table_name }
    }

    /// Create the indexes table if it doesn't exist.
    async fn create_table(&self) -> Result<()> {
        match &self.pool {
            DatabasePool::Postgres(pool) => {
                let query = format!(
                    r#"
                    CREATE TABLE IF NOT EXISTS {} (
                        index_id TEXT PRIMARY KEY,
                        index_type TEXT NOT NULL,
                        config JSONB NOT NULL,
                        node_ids JSONB NOT NULL,
                        metadata JSONB NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL
                    )
                    "#,
                    self.table_name
                );
                sqlx::query(&query).execute(pool).await.map_err(|e| {
                    CheungfunError::VectorStore {
                        message: format!("Failed to create indexes table: {}", e),
                    }
                })?;
            }
            DatabasePool::Sqlite(pool) => {
                let query = format!(
                    r#"
                    CREATE TABLE IF NOT EXISTS {} (
                        index_id TEXT PRIMARY KEY,
                        index_type TEXT NOT NULL,
                        config TEXT NOT NULL,
                        node_ids TEXT NOT NULL,
                        metadata TEXT NOT NULL,
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL
                    )
                    "#,
                    self.table_name
                );
                sqlx::query(&query).execute(pool).await.map_err(|e| {
                    CheungfunError::VectorStore {
                        message: format!("Failed to create indexes table: {}", e),
                    }
                })?;
            }
        }

        debug!("Created indexes table: {}", self.table_name);
        Ok(())
    }
}

#[async_trait]
impl IndexStore for SqlxIndexStore {
    async fn add_index_struct(&self, index_struct: IndexStruct) -> Result<()> {
        match &self.pool {
            DatabasePool::Postgres(pool) => {
                let config_json = serde_json::to_value(&index_struct.config).unwrap_or_default();
                let node_ids_json =
                    serde_json::to_value(&index_struct.node_ids).unwrap_or_default();
                let metadata_json =
                    serde_json::to_value(&index_struct.metadata).unwrap_or_default();

                let query = format!(
                    r#"
                    INSERT INTO {} (index_id, index_type, config, node_ids, metadata, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (index_id) DO UPDATE SET
                        index_type = EXCLUDED.index_type,
                        config = EXCLUDED.config,
                        node_ids = EXCLUDED.node_ids,
                        metadata = EXCLUDED.metadata,
                        updated_at = EXCLUDED.updated_at
                    "#,
                    self.table_name
                );

                sqlx::query(&query)
                    .bind(&index_struct.index_id)
                    .bind(&index_struct.index_type)
                    .bind(&config_json)
                    .bind(&node_ids_json)
                    .bind(&metadata_json)
                    .bind(&index_struct.created_at)
                    .bind(&index_struct.updated_at)
                    .execute(pool)
                    .await
                    .map_err(|e| CheungfunError::VectorStore {
                        message: format!("Failed to insert index {}: {}", index_struct.index_id, e),
                    })?;
            }
            DatabasePool::Sqlite(pool) => {
                let config_json = serde_json::to_string(&index_struct.config).unwrap_or_default();
                let node_ids_json =
                    serde_json::to_string(&index_struct.node_ids).unwrap_or_default();
                let metadata_json =
                    serde_json::to_string(&index_struct.metadata).unwrap_or_default();

                let query = format!(
                    r#"
                    INSERT OR REPLACE INTO {} (index_id, index_type, config, node_ids, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    "#,
                    self.table_name
                );

                sqlx::query(&query)
                    .bind(&index_struct.index_id)
                    .bind(&index_struct.index_type)
                    .bind(&config_json)
                    .bind(&node_ids_json)
                    .bind(&metadata_json)
                    .bind(&index_struct.created_at.to_rfc3339())
                    .bind(&index_struct.updated_at.to_rfc3339())
                    .execute(pool)
                    .await
                    .map_err(|e| CheungfunError::VectorStore {
                        message: format!("Failed to insert index {}: {}", index_struct.index_id, e),
                    })?;
            }
        }

        debug!("Added index structure: {}", index_struct.index_id);
        Ok(())
    }

    async fn get_index_struct(&self, struct_id: &str) -> Result<Option<IndexStruct>> {
        match &self.pool {
            DatabasePool::Postgres(pool) => {
                let query = format!("SELECT * FROM {} WHERE index_id = $1", self.table_name);

                let row = sqlx::query(&query)
                    .bind(struct_id)
                    .fetch_optional(pool)
                    .await
                    .map_err(|e| CheungfunError::VectorStore {
                        message: format!("Failed to fetch index {}: {}", struct_id, e),
                    })?;

                if let Some(row) = row {
                    let index_id: String = row.get("index_id");
                    let index_type: String = row.get("index_type");
                    let config_value: serde_json::Value = row.get("config");
                    let node_ids_value: serde_json::Value = row.get("node_ids");
                    let metadata_value: serde_json::Value = row.get("metadata");
                    let created_at: chrono::DateTime<chrono::Utc> = row.get("created_at");
                    let updated_at: chrono::DateTime<chrono::Utc> = row.get("updated_at");

                    let config = serde_json::from_value(config_value).unwrap_or_default();
                    let node_ids = serde_json::from_value(node_ids_value).unwrap_or_default();
                    let metadata = serde_json::from_value(metadata_value).unwrap_or_default();

                    Ok(Some(IndexStruct {
                        index_id,
                        index_type,
                        config,
                        node_ids,
                        metadata,
                        created_at,
                        updated_at,
                    }))
                } else {
                    Ok(None)
                }
            }
            DatabasePool::Sqlite(pool) => {
                let query = format!("SELECT * FROM {} WHERE index_id = ?", self.table_name);

                let row = sqlx::query(&query)
                    .bind(struct_id)
                    .fetch_optional(pool)
                    .await
                    .map_err(|e| CheungfunError::VectorStore {
                        message: format!("Failed to fetch index {}: {}", struct_id, e),
                    })?;

                if let Some(row) = row {
                    let index_id: String = row.get("index_id");
                    let index_type: String = row.get("index_type");
                    let config_str: String = row.get("config");
                    let node_ids_str: String = row.get("node_ids");
                    let metadata_str: String = row.get("metadata");
                    let created_at_str: String = row.get("created_at");
                    let updated_at_str: String = row.get("updated_at");

                    let config = serde_json::from_str(&config_str).unwrap_or_default();
                    let node_ids = serde_json::from_str(&node_ids_str).unwrap_or_default();
                    let metadata = serde_json::from_str(&metadata_str).unwrap_or_default();

                    let created_at = chrono::DateTime::parse_from_rfc3339(&created_at_str)
                        .unwrap_or_else(|_| chrono::Utc::now().into())
                        .with_timezone(&chrono::Utc);
                    let updated_at = chrono::DateTime::parse_from_rfc3339(&updated_at_str)
                        .unwrap_or_else(|_| chrono::Utc::now().into())
                        .with_timezone(&chrono::Utc);

                    Ok(Some(IndexStruct {
                        index_id,
                        index_type,
                        config,
                        node_ids,
                        metadata,
                        created_at,
                        updated_at,
                    }))
                } else {
                    Ok(None)
                }
            }
        }
    }

    async fn delete_index_struct(&self, struct_id: &str) -> Result<()> {
        match &self.pool {
            DatabasePool::Postgres(pool) => {
                let query = format!("DELETE FROM {} WHERE index_id = $1", self.table_name);
                sqlx::query(&query)
                    .bind(struct_id)
                    .execute(pool)
                    .await
                    .map_err(|e| CheungfunError::VectorStore {
                        message: format!("Failed to delete index {}: {}", struct_id, e),
                    })?;
            }
            DatabasePool::Sqlite(pool) => {
                let query = format!("DELETE FROM {} WHERE index_id = ?", self.table_name);
                sqlx::query(&query)
                    .bind(struct_id)
                    .execute(pool)
                    .await
                    .map_err(|e| CheungfunError::VectorStore {
                        message: format!("Failed to delete index {}: {}", struct_id, e),
                    })?;
            }
        }

        debug!("Deleted index structure: {}", struct_id);
        Ok(())
    }

    async fn list_index_structs(&self) -> Result<Vec<String>> {
        let mut index_ids = Vec::new();

        match &self.pool {
            DatabasePool::Postgres(pool) => {
                let query = format!("SELECT index_id FROM {}", self.table_name);
                let rows = sqlx::query(&query).fetch_all(pool).await.map_err(|e| {
                    CheungfunError::VectorStore {
                        message: format!("Failed to list indexes: {}", e),
                    }
                })?;

                for row in rows {
                    let index_id: String = row.get("index_id");
                    index_ids.push(index_id);
                }
            }
            DatabasePool::Sqlite(pool) => {
                let query = format!("SELECT index_id FROM {}", self.table_name);
                let rows = sqlx::query(&query).fetch_all(pool).await.map_err(|e| {
                    CheungfunError::VectorStore {
                        message: format!("Failed to list indexes: {}", e),
                    }
                })?;

                for row in rows {
                    let index_id: String = row.get("index_id");
                    index_ids.push(index_id);
                }
            }
        }

        Ok(index_ids)
    }

    async fn clear(&self) -> Result<()> {
        match &self.pool {
            DatabasePool::Postgres(pool) => {
                let query = format!("DELETE FROM {}", self.table_name);
                sqlx::query(&query).execute(pool).await.map_err(|e| {
                    CheungfunError::VectorStore {
                        message: format!("Failed to clear indexes: {}", e),
                    }
                })?;
            }
            DatabasePool::Sqlite(pool) => {
                let query = format!("DELETE FROM {}", self.table_name);
                sqlx::query(&query).execute(pool).await.map_err(|e| {
                    CheungfunError::VectorStore {
                        message: format!("Failed to clear indexes: {}", e),
                    }
                })?;
            }
        }

        info!("Cleared all index structures from store");
        Ok(())
    }
}
