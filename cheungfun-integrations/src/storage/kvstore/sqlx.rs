//! SQLx-based KVStore implementation.

use async_trait::async_trait;
use cheungfun_core::{traits::KVStore, CheungfunError, Result};
use serde_json::Value;
use sqlx::{Pool, Postgres, Row, Sqlite};
use std::collections::HashMap;
use tracing::{debug, error, info};

/// Database pool enum supporting both PostgreSQL and SQLite.
#[derive(Debug, Clone)]
pub enum DatabasePool {
    /// PostgreSQL connection pool.
    Postgres(Pool<Postgres>),
    /// SQLite connection pool.
    Sqlite(Pool<Sqlite>),
}

/// SQLx-based key-value store implementation.
///
/// This store uses SQLx to provide persistent storage with support for
/// both PostgreSQL and SQLite backends. It creates a unified table
/// structure for storing key-value pairs with collection-based organization.
///
/// # Table Structure
///
/// The store creates a table with the following structure:
/// - `collection`: TEXT - The collection/namespace name
/// - `key`: TEXT - The key within the collection
/// - `value`: JSONB/TEXT - The stored value (JSONB for PostgreSQL, TEXT for SQLite)
/// - `created_at`: TIMESTAMP - When the record was created
/// - `updated_at`: TIMESTAMP - When the record was last updated
///
/// # Examples
///
/// ```rust,no_run
/// use cheungfun_integrations::storage::kvstore::SqlxKVStore;
/// use cheungfun_core::traits::{KVStore, DEFAULT_COLLECTION};
/// use serde_json::json;
/// use sqlx::SqlitePool;
///
/// # tokio_test::block_on(async {
/// let pool = SqlitePool::connect(":memory:").await.unwrap();
/// let store = SqlxKVStore::new(pool.into(), "test_").await.unwrap();
///
/// // Store a value
/// store.put("key1", json!({"name": "test"}), DEFAULT_COLLECTION).await.unwrap();
///
/// // Retrieve the value
/// let value = store.get("key1", DEFAULT_COLLECTION).await.unwrap();
/// assert!(value.is_some());
/// # });
/// ```
#[derive(Debug)]
pub struct SqlxKVStore {
    /// Database connection pool.
    pool: DatabasePool,
    /// Table name for storing key-value pairs.
    table_name: String,
}

impl From<Pool<Postgres>> for DatabasePool {
    fn from(pool: Pool<Postgres>) -> Self {
        DatabasePool::Postgres(pool)
    }
}

impl From<Pool<Sqlite>> for DatabasePool {
    fn from(pool: Pool<Sqlite>) -> Self {
        DatabasePool::Sqlite(pool)
    }
}

impl SqlxKVStore {
    /// Create a new SQLx KV store with the given pool and table prefix.
    ///
    /// # Arguments
    ///
    /// * `pool` - Database connection pool
    /// * `table_prefix` - Prefix for the table name (e.g., "cheungfun_")
    ///
    /// # Errors
    ///
    /// Returns an error if table creation fails.
    pub async fn new(pool: DatabasePool, table_prefix: &str) -> Result<Self> {
        let table_name = format!("{}kv_store", table_prefix);
        let store = Self { pool, table_name };
        store.create_table().await?;
        info!("Created SQLx KV store with table '{}'", store.table_name);
        Ok(store)
    }

    /// Create the KV store table if it doesn't exist.
    async fn create_table(&self) -> Result<()> {
        match &self.pool {
            DatabasePool::Postgres(pool) => {
                let query = format!(
                    r#"
                    CREATE TABLE IF NOT EXISTS {} (
                        collection TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value JSONB NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW(),
                        PRIMARY KEY (collection, key)
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_{}_collection 
                    ON {} (collection);
                    
                    CREATE INDEX IF NOT EXISTS idx_{}_updated_at 
                    ON {} (updated_at);
                    "#,
                    self.table_name,
                    self.table_name.replace('.', "_"),
                    self.table_name,
                    self.table_name.replace('.', "_"),
                    self.table_name
                );
                sqlx::query(&query).execute(pool).await?;
            }
            DatabasePool::Sqlite(pool) => {
                let query = format!(
                    r#"
                    CREATE TABLE IF NOT EXISTS {} (
                        collection TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (collection, key)
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_{}_collection 
                    ON {} (collection);
                    
                    CREATE INDEX IF NOT EXISTS idx_{}_updated_at 
                    ON {} (updated_at);
                    "#,
                    self.table_name,
                    self.table_name.replace('.', "_"),
                    self.table_name,
                    self.table_name.replace('.', "_"),
                    self.table_name
                );
                sqlx::query(&query).execute(pool).await?;
            }
        }
        debug!("Created table '{}' with indexes", self.table_name);
        Ok(())
    }

    /// Get database-specific current timestamp expression.
    fn current_timestamp(&self) -> &'static str {
        match &self.pool {
            DatabasePool::Postgres(_) => "NOW()",
            DatabasePool::Sqlite(_) => "CURRENT_TIMESTAMP",
        }
    }

    /// Get the database type name.
    pub fn database_type(&self) -> &'static str {
        match &self.pool {
            DatabasePool::Postgres(_) => "PostgreSQL",
            DatabasePool::Sqlite(_) => "SQLite",
        }
    }

    /// Get connection pool statistics.
    pub fn pool_stats(&self) -> PoolStats {
        match &self.pool {
            DatabasePool::Postgres(pool) => PoolStats {
                size: pool.size(),
                idle: pool.num_idle(),
                database_type: "PostgreSQL".to_string(),
            },
            DatabasePool::Sqlite(pool) => PoolStats {
                size: pool.size(),
                idle: pool.num_idle(),
                database_type: "SQLite".to_string(),
            },
        }
    }
}

#[async_trait]
impl KVStore for SqlxKVStore {
    async fn put(&self, key: &str, value: Value, collection: &str) -> Result<()> {
        match &self.pool {
            DatabasePool::Postgres(pool) => {
                let query = format!(
                    r#"
                    INSERT INTO {} (collection, key, value, created_at, updated_at)
                    VALUES ($1, $2, $3, NOW(), NOW())
                    ON CONFLICT (collection, key) 
                    DO UPDATE SET 
                        value = EXCLUDED.value, 
                        updated_at = NOW()
                    "#,
                    self.table_name
                );
                sqlx::query(&query)
                    .bind(collection)
                    .bind(key)
                    .bind(&value)
                    .execute(pool)
                    .await
                    .map_err(|e| {
                        error!(
                            "Failed to put key '{}' in collection '{}': {}",
                            key, collection, e
                        );
                        CheungfunError::Storage(format!("Database error: {}", e))
                    })?;
            }
            DatabasePool::Sqlite(pool) => {
                let value_str =
                    serde_json::to_string(&value).map_err(|e| CheungfunError::Serialization(e))?;

                let query = format!(
                    r#"
                    INSERT OR REPLACE INTO {} (collection, key, value, created_at, updated_at)
                    VALUES (?, ?, ?, 
                        COALESCE((SELECT created_at FROM {} WHERE collection = ? AND key = ?), CURRENT_TIMESTAMP),
                        CURRENT_TIMESTAMP)
                    "#,
                    self.table_name, self.table_name
                );
                sqlx::query(&query)
                    .bind(collection)
                    .bind(key)
                    .bind(&value_str)
                    .bind(collection)
                    .bind(key)
                    .execute(pool)
                    .await
                    .map_err(|e| {
                        error!(
                            "Failed to put key '{}' in collection '{}': {}",
                            key, collection, e
                        );
                        CheungfunError::Storage(format!("Database error: {}", e))
                    })?;
            }
        }

        debug!("Put key '{}' in collection '{}'", key, collection);
        Ok(())
    }

    async fn get(&self, key: &str, collection: &str) -> Result<Option<Value>> {
        match &self.pool {
            DatabasePool::Postgres(pool) => {
                let query = format!(
                    "SELECT value FROM {} WHERE collection = $1 AND key = $2",
                    self.table_name
                );
                let row = sqlx::query(&query)
                    .bind(collection)
                    .bind(key)
                    .fetch_optional(pool)
                    .await
                    .map_err(|e| {
                        error!(
                            "Failed to get key '{}' from collection '{}': {}",
                            key, collection, e
                        );
                        CheungfunError::Storage(format!("Database error: {}", e))
                    })?;

                if let Some(row) = row {
                    let value: Value = row.get("value");
                    debug!("Get key '{}' from collection '{}': found", key, collection);
                    Ok(Some(value))
                } else {
                    debug!(
                        "Get key '{}' from collection '{}': not found",
                        key, collection
                    );
                    Ok(None)
                }
            }
            DatabasePool::Sqlite(pool) => {
                let query = format!(
                    "SELECT value FROM {} WHERE collection = ? AND key = ?",
                    self.table_name
                );
                let row = sqlx::query(&query)
                    .bind(collection)
                    .bind(key)
                    .fetch_optional(pool)
                    .await
                    .map_err(|e| {
                        error!(
                            "Failed to get key '{}' from collection '{}': {}",
                            key, collection, e
                        );
                        CheungfunError::Storage(format!("Database error: {}", e))
                    })?;

                if let Some(row) = row {
                    let value_str: String = row.get("value");
                    let value: Value = serde_json::from_str(&value_str)
                        .map_err(|e| CheungfunError::Serialization(e))?;
                    debug!("Get key '{}' from collection '{}': found", key, collection);
                    Ok(Some(value))
                } else {
                    debug!(
                        "Get key '{}' from collection '{}': not found",
                        key, collection
                    );
                    Ok(None)
                }
            }
        }
    }

    async fn delete(&self, key: &str, collection: &str) -> Result<bool> {
        let query = format!(
            "DELETE FROM {} WHERE collection = {} AND key = {}",
            self.table_name,
            match &self.pool {
                DatabasePool::Postgres(_) => "$1",
                DatabasePool::Sqlite(_) => "?",
            },
            match &self.pool {
                DatabasePool::Postgres(_) => "$2",
                DatabasePool::Sqlite(_) => "?",
            }
        );

        let rows_affected = match &self.pool {
            DatabasePool::Postgres(pool) => sqlx::query(&query)
                .bind(collection)
                .bind(key)
                .execute(pool)
                .await?
                .rows_affected(),
            DatabasePool::Sqlite(pool) => sqlx::query(&query)
                .bind(collection)
                .bind(key)
                .execute(pool)
                .await?
                .rows_affected(),
        };

        let deleted = rows_affected > 0;
        debug!(
            "Delete key '{}' from collection '{}': {}",
            key,
            collection,
            if deleted { "deleted" } else { "not found" }
        );
        Ok(deleted)
    }

    async fn get_all(&self, collection: &str) -> Result<HashMap<String, Value>> {
        let query = format!(
            "SELECT key, value FROM {} WHERE collection = {}",
            self.table_name,
            match &self.pool {
                DatabasePool::Postgres(_) => "$1",
                DatabasePool::Sqlite(_) => "?",
            }
        );

        match &self.pool {
            DatabasePool::Postgres(pool) => {
                let rows = sqlx::query(&query).bind(collection).fetch_all(pool).await?;

                let mut result = HashMap::new();
                for row in rows {
                    let key: String = row.get("key");
                    let value_str: String = row.get("value");
                    let value: Value = serde_json::from_str(&value_str)
                        .map_err(|e| CheungfunError::Serialization(e))?;
                    result.insert(key, value);
                }
                debug!(
                    "Get all from collection '{}': {} items",
                    collection,
                    result.len()
                );
                Ok(result)
            }
            DatabasePool::Sqlite(pool) => {
                let rows = sqlx::query(&query).bind(collection).fetch_all(pool).await?;

                let mut result = HashMap::new();
                for row in rows {
                    let key: String = row.get("key");
                    let value_str: String = row.get("value");
                    let value: Value = serde_json::from_str(&value_str)
                        .map_err(|e| CheungfunError::Serialization(e))?;
                    result.insert(key, value);
                }
                debug!(
                    "Get all from collection '{}': {} items",
                    collection,
                    result.len()
                );
                Ok(result)
            }
        }
    }

    async fn list_collections(&self) -> Result<Vec<String>> {
        let query = format!("SELECT DISTINCT collection FROM {}", self.table_name);

        match &self.pool {
            DatabasePool::Postgres(pool) => {
                let rows = sqlx::query(&query).fetch_all(pool).await?;
                let mut collections = Vec::new();
                for row in rows {
                    let collection: String = row.get("collection");
                    collections.push(collection);
                }
                debug!("List collections: {} found", collections.len());
                Ok(collections)
            }
            DatabasePool::Sqlite(pool) => {
                let rows = sqlx::query(&query).fetch_all(pool).await?;
                let mut collections = Vec::new();
                for row in rows {
                    let collection: String = row.get("collection");
                    collections.push(collection);
                }
                debug!("List collections: {} found", collections.len());
                Ok(collections)
            }
        }
    }

    async fn delete_collection(&self, collection: &str) -> Result<()> {
        let query = format!(
            "DELETE FROM {} WHERE collection = {}",
            self.table_name,
            match &self.pool {
                DatabasePool::Postgres(_) => "$1",
                DatabasePool::Sqlite(_) => "?",
            }
        );

        let rows_affected = match &self.pool {
            DatabasePool::Postgres(pool) => sqlx::query(&query)
                .bind(collection)
                .execute(pool)
                .await?
                .rows_affected(),
            DatabasePool::Sqlite(pool) => sqlx::query(&query)
                .bind(collection)
                .execute(pool)
                .await?
                .rows_affected(),
        };

        debug!(
            "Deleted collection '{}': {} rows affected",
            collection, rows_affected
        );
        Ok(())
    }

    async fn count(&self, collection: &str) -> Result<usize> {
        let query = format!(
            "SELECT COUNT(*) as count FROM {} WHERE collection = {}",
            self.table_name,
            match &self.pool {
                DatabasePool::Postgres(_) => "$1",
                DatabasePool::Sqlite(_) => "?",
            }
        );

        let count = match &self.pool {
            DatabasePool::Postgres(pool) => {
                let row = sqlx::query(&query).bind(collection).fetch_one(pool).await?;
                let count: i64 = row.get("count");
                count as usize
            }
            DatabasePool::Sqlite(pool) => {
                let row = sqlx::query(&query).bind(collection).fetch_one(pool).await?;
                let count: i64 = row.get("count");
                count as usize
            }
        };

        debug!("Count collection '{}': {} items", collection, count);
        Ok(count)
    }

    fn name(&self) -> &'static str {
        "SqlxKVStore"
    }
}

/// Connection pool statistics.
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Total pool size.
    pub size: u32,
    /// Number of idle connections.
    pub idle: usize,
    /// Database type.
    pub database_type: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use sqlx::SqlitePool;

    async fn create_test_store() -> SqlxKVStore {
        let pool = SqlitePool::connect(":memory:").await.unwrap();
        SqlxKVStore::new(pool.into(), "test_").await.unwrap()
    }

    #[tokio::test]
    async fn test_basic_operations() {
        let store = create_test_store().await;
        let collection = "test";

        // Test put and get
        store
            .put("key1", json!({"value": 42}), collection)
            .await
            .unwrap();
        let result = store.get("key1", collection).await.unwrap();
        assert_eq!(result, Some(json!({"value": 42})));

        // Test non-existent key
        let result = store.get("nonexistent", collection).await.unwrap();
        assert_eq!(result, None);

        // Test delete
        let deleted = store.delete("key1", collection).await.unwrap();
        assert!(deleted);

        let result = store.get("key1", collection).await.unwrap();
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn test_collections() {
        let store = create_test_store().await;

        // Add data to different collections
        store
            .put("key1", json!("value1"), "collection1")
            .await
            .unwrap();
        store
            .put("key1", json!("value2"), "collection2")
            .await
            .unwrap();

        // Verify isolation
        let val1 = store.get("key1", "collection1").await.unwrap();
        let val2 = store.get("key1", "collection2").await.unwrap();

        assert_eq!(val1, Some(json!("value1")));
        assert_eq!(val2, Some(json!("value2")));

        // Test list collections
        let collections = store.list_collections().await.unwrap();
        assert_eq!(collections.len(), 2);
        assert!(collections.contains(&"collection1".to_string()));
        assert!(collections.contains(&"collection2".to_string()));
    }

    #[tokio::test]
    async fn test_count_and_get_all() {
        let store = create_test_store().await;
        let collection = "test";

        // Add some data
        store
            .put("key1", json!("value1"), collection)
            .await
            .unwrap();
        store
            .put("key2", json!("value2"), collection)
            .await
            .unwrap();
        store
            .put("key3", json!("value3"), collection)
            .await
            .unwrap();

        // Test count
        let count = store.count(collection).await.unwrap();
        assert_eq!(count, 3);

        // Test get_all
        let all_data = store.get_all(collection).await.unwrap();
        assert_eq!(all_data.len(), 3);
        assert_eq!(all_data.get("key1"), Some(&json!("value1")));
        assert_eq!(all_data.get("key2"), Some(&json!("value2")));
        assert_eq!(all_data.get("key3"), Some(&json!("value3")));
    }
}
