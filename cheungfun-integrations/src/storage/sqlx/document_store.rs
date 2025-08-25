//! SQLx-based document store implementation.

use async_trait::async_trait;
use cheungfun_core::{traits::DocumentStore, CheungfunError, Document, Result};
use sqlx::Row;
use std::collections::HashMap;
use tracing::{debug, info};

use super::{DatabasePool, SqlxStorageConfig};

/// SQLx-based document store implementation.
///
/// This store persists documents in a relational database using SQLx,
/// supporting both PostgreSQL and SQLite backends.
#[derive(Debug)]
pub struct SqlxDocumentStore {
    pool: DatabasePool,
    table_name: String,
}

impl SqlxDocumentStore {
    /// Create a new SQLx document store.
    pub async fn new(config: SqlxStorageConfig) -> Result<Self> {
        let pool = config
            .create_pool()
            .await
            .map_err(|e| CheungfunError::Configuration {
                message: format!("Failed to create database pool: {}", e),
            })?;

        let table_name = format!("{}documents", config.table_prefix);

        let store = Self {
            pool,
            table_name: table_name.clone(),
        };

        if config.auto_migrate {
            store.create_table().await?;
        }

        info!("Created SQLx document store with table: {}", table_name);
        Ok(store)
    }

    /// Create a new document store with an existing pool.
    pub fn with_pool(pool: DatabasePool, table_prefix: &str) -> Self {
        let table_name = format!("{}documents", table_prefix);
        Self { pool, table_name }
    }

    /// Create the documents table if it doesn't exist.
    async fn create_table(&self) -> Result<()> {
        match &self.pool {
            DatabasePool::Postgres(pool) => {
                let query = format!(
                    r#"
                    CREATE TABLE IF NOT EXISTS {} (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        metadata JSONB,
                        content_hash TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                    "#,
                    self.table_name
                );
                sqlx::query(&query).execute(pool).await.map_err(|e| {
                    CheungfunError::VectorStore {
                        message: format!("Failed to create documents table: {}", e),
                    }
                })?;
            }
            DatabasePool::Sqlite(pool) => {
                let query = format!(
                    r#"
                    CREATE TABLE IF NOT EXISTS {} (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        metadata TEXT,
                        content_hash TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                    "#,
                    self.table_name
                );
                sqlx::query(&query).execute(pool).await.map_err(|e| {
                    CheungfunError::VectorStore {
                        message: format!("Failed to create documents table: {}", e),
                    }
                })?;
            }
        }

        debug!("Created documents table: {}", self.table_name);
        Ok(())
    }

    /// Calculate content hash for change detection.
    fn calculate_hash(content: &str) -> String {
        // Simple hash implementation using std library
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

#[async_trait]
impl DocumentStore for SqlxDocumentStore {
    async fn add_documents(&self, docs: Vec<Document>) -> Result<Vec<String>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }

        let mut doc_ids = Vec::with_capacity(docs.len());

        match &self.pool {
            DatabasePool::Postgres(pool) => {
                for doc in docs {
                    let doc_id = doc.id.clone();
                    let content_hash = Self::calculate_hash(&doc.content);
                    let metadata_json = serde_json::to_value(&doc.metadata).unwrap_or_default();

                    let query = format!(
                        r#"
                        INSERT INTO {} (id, content, metadata, content_hash)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (id) DO UPDATE SET
                            content = EXCLUDED.content,
                            metadata = EXCLUDED.metadata,
                            content_hash = EXCLUDED.content_hash,
                            updated_at = NOW()
                        "#,
                        self.table_name
                    );

                    sqlx::query(&query)
                        .bind(&doc_id.to_string())
                        .bind(&doc.content)
                        .bind(&metadata_json)
                        .bind(&content_hash)
                        .execute(pool)
                        .await
                        .map_err(|e| CheungfunError::VectorStore {
                            message: format!("Failed to insert document {}: {}", doc_id, e),
                        })?;

                    doc_ids.push(doc_id.to_string());
                }
            }
            DatabasePool::Sqlite(pool) => {
                for doc in docs {
                    let doc_id = doc.id.clone();
                    let content_hash = Self::calculate_hash(&doc.content);
                    let metadata_json = serde_json::to_string(&doc.metadata).unwrap_or_default();

                    let query = format!(
                        r#"
                        INSERT OR REPLACE INTO {} (id, content, metadata, content_hash)
                        VALUES (?, ?, ?, ?)
                        "#,
                        self.table_name
                    );

                    sqlx::query(&query)
                        .bind(&doc_id.to_string())
                        .bind(&doc.content)
                        .bind(&metadata_json)
                        .bind(&content_hash)
                        .execute(pool)
                        .await
                        .map_err(|e| CheungfunError::VectorStore {
                            message: format!("Failed to insert document {}: {}", doc_id, e),
                        })?;

                    doc_ids.push(doc_id.to_string());
                }
            }
        }

        debug!("Added {} documents to store", doc_ids.len());
        Ok(doc_ids)
    }

    async fn get_document(&self, doc_id: &str) -> Result<Option<Document>> {
        match &self.pool {
            DatabasePool::Postgres(pool) => {
                let query = format!(
                    "SELECT id, content, metadata FROM {} WHERE id = $1",
                    self.table_name
                );

                let row = sqlx::query(&query)
                    .bind(doc_id)
                    .fetch_optional(pool)
                    .await
                    .map_err(|e| CheungfunError::VectorStore {
                        message: format!("Failed to fetch document {}: {}", doc_id, e),
                    })?;

                if let Some(row) = row {
                    let id: String = row.get("id");
                    let content: String = row.get("content");
                    let metadata_value: serde_json::Value = row.get("metadata");
                    let metadata: HashMap<String, serde_json::Value> =
                        serde_json::from_value(metadata_value).unwrap_or_default();

                    Ok(Some(Document {
                        id: uuid::Uuid::parse_str(&id).map_err(|e| {
                            CheungfunError::validation(format!("Invalid UUID format: {}", e))
                        })?,
                        content,
                        metadata,
                        embedding: None, // TODO: Load embedding from database
                    }))
                } else {
                    Ok(None)
                }
            }
            DatabasePool::Sqlite(pool) => {
                let query = format!(
                    "SELECT id, content, metadata FROM {} WHERE id = ?",
                    self.table_name
                );

                let row = sqlx::query(&query)
                    .bind(doc_id)
                    .fetch_optional(pool)
                    .await
                    .map_err(|e| CheungfunError::VectorStore {
                        message: format!("Failed to fetch document {}: {}", doc_id, e),
                    })?;

                if let Some(row) = row {
                    let id: String = row.get("id");
                    let content: String = row.get("content");
                    let metadata_str: String = row.get("metadata");
                    let metadata: HashMap<String, serde_json::Value> =
                        serde_json::from_str(&metadata_str).unwrap_or_default();

                    Ok(Some(Document {
                        id: uuid::Uuid::parse_str(&id).map_err(|e| {
                            CheungfunError::validation(format!("Invalid UUID format: {}", e))
                        })?,
                        content,
                        metadata,
                        embedding: None, // TODO: Load embedding from database
                    }))
                } else {
                    Ok(None)
                }
            }
        }
    }

    async fn get_documents(&self, doc_ids: Vec<String>) -> Result<Vec<Document>> {
        if doc_ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut documents = Vec::new();

        for doc_id in doc_ids {
            if let Some(doc) = self.get_document(&doc_id).await? {
                documents.push(doc);
            }
        }

        Ok(documents)
    }

    async fn delete_document(&self, doc_id: &str) -> Result<()> {
        match &self.pool {
            DatabasePool::Postgres(pool) => {
                let query = format!("DELETE FROM {} WHERE id = $1", self.table_name);
                sqlx::query(&query)
                    .bind(doc_id)
                    .execute(pool)
                    .await
                    .map_err(|e| CheungfunError::VectorStore {
                        message: format!("Failed to delete document {}: {}", doc_id, e),
                    })?;
            }
            DatabasePool::Sqlite(pool) => {
                let query = format!("DELETE FROM {} WHERE id = ?", self.table_name);
                sqlx::query(&query)
                    .bind(doc_id)
                    .execute(pool)
                    .await
                    .map_err(|e| CheungfunError::VectorStore {
                        message: format!("Failed to delete document {}: {}", doc_id, e),
                    })?;
            }
        }

        debug!("Deleted document: {}", doc_id);
        Ok(())
    }

    async fn get_all_document_hashes(&self) -> Result<HashMap<String, String>> {
        let mut hashes = HashMap::new();

        match &self.pool {
            DatabasePool::Postgres(pool) => {
                let query = format!("SELECT id, content_hash FROM {}", self.table_name);
                let rows = sqlx::query(&query).fetch_all(pool).await.map_err(|e| {
                    CheungfunError::VectorStore {
                        message: format!("Failed to fetch document hashes: {}", e),
                    }
                })?;

                for row in rows {
                    let id: String = row.get("id");
                    let hash: String = row.get("content_hash");
                    hashes.insert(id, hash);
                }
            }
            DatabasePool::Sqlite(pool) => {
                let query = format!("SELECT id, content_hash FROM {}", self.table_name);
                let rows = sqlx::query(&query).fetch_all(pool).await.map_err(|e| {
                    CheungfunError::VectorStore {
                        message: format!("Failed to fetch document hashes: {}", e),
                    }
                })?;

                for row in rows {
                    let id: String = row.get("id");
                    let hash: String = row.get("content_hash");
                    hashes.insert(id, hash);
                }
            }
        }

        Ok(hashes)
    }

    async fn count_documents(&self) -> Result<usize> {
        match &self.pool {
            DatabasePool::Postgres(pool) => {
                let query = format!("SELECT COUNT(*) as count FROM {}", self.table_name);
                let row = sqlx::query(&query).fetch_one(pool).await.map_err(|e| {
                    CheungfunError::VectorStore {
                        message: format!("Failed to count documents: {}", e),
                    }
                })?;
                let count: i64 = row.get("count");
                Ok(count as usize)
            }
            DatabasePool::Sqlite(pool) => {
                let query = format!("SELECT COUNT(*) as count FROM {}", self.table_name);
                let row = sqlx::query(&query).fetch_one(pool).await.map_err(|e| {
                    CheungfunError::VectorStore {
                        message: format!("Failed to count documents: {}", e),
                    }
                })?;
                let count: i64 = row.get("count");
                Ok(count as usize)
            }
        }
    }

    async fn clear(&self) -> Result<()> {
        match &self.pool {
            DatabasePool::Postgres(pool) => {
                let query = format!("DELETE FROM {}", self.table_name);
                sqlx::query(&query).execute(pool).await.map_err(|e| {
                    CheungfunError::VectorStore {
                        message: format!("Failed to clear documents: {}", e),
                    }
                })?;
            }
            DatabasePool::Sqlite(pool) => {
                let query = format!("DELETE FROM {}", self.table_name);
                sqlx::query(&query).execute(pool).await.map_err(|e| {
                    CheungfunError::VectorStore {
                        message: format!("Failed to clear documents: {}", e),
                    }
                })?;
            }
        }

        info!("Cleared all documents from store");
        Ok(())
    }
}
