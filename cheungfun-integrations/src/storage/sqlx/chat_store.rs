//! SQLx-based chat store implementation.

use async_trait::async_trait;
use cheungfun_core::{traits::ChatStore, ChatMessage, CheungfunError, Result};
use sqlx::Row;
use tracing::{debug, info};

use super::{DatabasePool, SqlxStorageConfig};

/// SQLx-based chat store implementation.
///
/// This store persists conversation history in a relational database using SQLx,
/// supporting both PostgreSQL and SQLite backends.
#[derive(Debug)]
pub struct SqlxChatStore {
    pool: DatabasePool,
    table_name: String,
}

impl SqlxChatStore {
    /// Create a new SQLx chat store.
    pub async fn new(config: SqlxStorageConfig) -> Result<Self> {
        let pool = config
            .create_pool()
            .await
            .map_err(|e| CheungfunError::Configuration {
                message: format!("Failed to create database pool: {}", e),
            })?;

        let table_name = format!("{}conversations", config.table_prefix);

        let store = Self {
            pool,
            table_name: table_name.clone(),
        };

        if config.auto_migrate {
            store.create_table().await?;
        }

        info!("Created SQLx chat store with table: {}", table_name);
        Ok(store)
    }

    /// Create a new chat store with an existing pool.
    pub fn with_pool(pool: DatabasePool, table_prefix: &str) -> Self {
        let table_name = format!("{}conversations", table_prefix);
        Self { pool, table_name }
    }

    /// Create the conversations table if it doesn't exist.
    async fn create_table(&self) -> Result<()> {
        match &self.pool {
            DatabasePool::Postgres(pool) => {
                let query = format!(
                    r#"
                    CREATE TABLE IF NOT EXISTS {} (
                        id SERIAL PRIMARY KEY,
                        conversation_key TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata JSONB,
                        message_order INTEGER NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    CREATE INDEX IF NOT EXISTS idx_{}_conversation_key ON {} (conversation_key);
                    CREATE INDEX IF NOT EXISTS idx_{}_order ON {} (conversation_key, message_order);
                    "#,
                    self.table_name,
                    self.table_name.replace(".", "_"),
                    self.table_name,
                    self.table_name.replace(".", "_"),
                    self.table_name
                );
                sqlx::query(&query).execute(pool).await.map_err(|e| {
                    CheungfunError::VectorStore {
                        message: format!("Failed to create conversations table: {}", e),
                    }
                })?;
            }
            DatabasePool::Sqlite(pool) => {
                let query = format!(
                    r#"
                    CREATE TABLE IF NOT EXISTS {} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_key TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata TEXT,
                        message_order INTEGER NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_{}_conversation_key ON {} (conversation_key);
                    CREATE INDEX IF NOT EXISTS idx_{}_order ON {} (conversation_key, message_order);
                    "#,
                    self.table_name,
                    self.table_name.replace(".", "_"),
                    self.table_name,
                    self.table_name.replace(".", "_"),
                    self.table_name
                );
                sqlx::query(&query).execute(pool).await.map_err(|e| {
                    CheungfunError::VectorStore {
                        message: format!("Failed to create conversations table: {}", e),
                    }
                })?;
            }
        }

        debug!("Created conversations table: {}", self.table_name);
        Ok(())
    }
}

#[async_trait]
impl ChatStore for SqlxChatStore {
    async fn set_messages(&self, key: &str, messages: Vec<ChatMessage>) -> Result<()> {
        // First, delete existing messages for this key
        self.delete_messages(key).await?;

        if messages.is_empty() {
            return Ok(());
        }

        let message_count = messages.len();

        // Then insert new messages
        match &self.pool {
            DatabasePool::Postgres(pool) => {
                for (order, message) in messages.iter().enumerate() {
                    let metadata_json = message
                        .metadata
                        .as_ref()
                        .map(|m| serde_json::to_value(m).unwrap_or_default())
                        .unwrap_or_default();

                    let query = format!(
                        r#"
                        INSERT INTO {} (conversation_key, role, content, metadata, message_order)
                        VALUES ($1, $2, $3, $4, $5)
                        "#,
                        self.table_name
                    );

                    sqlx::query(&query)
                        .bind(key)
                        .bind(format!("{:?}", message.role))
                        .bind(&message.content)
                        .bind(&metadata_json)
                        .bind(order as i32)
                        .execute(pool)
                        .await
                        .map_err(|e| CheungfunError::VectorStore {
                            message: format!("Failed to insert message: {}", e),
                        })?;
                }
            }
            DatabasePool::Sqlite(pool) => {
                for (order, message) in messages.iter().enumerate() {
                    let metadata_json = message
                        .metadata
                        .as_ref()
                        .map(|m| serde_json::to_string(m).unwrap_or_default())
                        .unwrap_or_default();

                    let query = format!(
                        r#"
                        INSERT INTO {} (conversation_key, role, content, metadata, message_order)
                        VALUES (?, ?, ?, ?, ?)
                        "#,
                        self.table_name
                    );

                    sqlx::query(&query)
                        .bind(key)
                        .bind(format!("{:?}", message.role))
                        .bind(&message.content)
                        .bind(&metadata_json)
                        .bind(order as i32)
                        .execute(pool)
                        .await
                        .map_err(|e| CheungfunError::VectorStore {
                            message: format!("Failed to insert message: {}", e),
                        })?;
                }
            }
        }

        debug!("Set {} messages for conversation: {}", message_count, key);
        Ok(())
    }

    async fn get_messages(&self, key: &str) -> Result<Vec<ChatMessage>> {
        let mut messages = Vec::new();

        match &self.pool {
            DatabasePool::Postgres(pool) => {
                let query = format!(
                    r#"
                    SELECT role, content, metadata, created_at
                    FROM {} 
                    WHERE conversation_key = $1 
                    ORDER BY message_order ASC
                    "#,
                    self.table_name
                );

                let rows = sqlx::query(&query)
                    .bind(key)
                    .fetch_all(pool)
                    .await
                    .map_err(|e| CheungfunError::VectorStore {
                        message: format!("Failed to fetch messages for {}: {}", key, e),
                    })?;

                for row in rows {
                    let role_str: String = row.get("role");
                    let content: String = row.get("content");
                    let metadata_value: serde_json::Value = row.get("metadata");
                    let timestamp: chrono::DateTime<chrono::Utc> = row.get("created_at");

                    let role = match role_str.as_str() {
                        "User" => cheungfun_core::MessageRole::User,
                        "Assistant" => cheungfun_core::MessageRole::Assistant,
                        "System" => cheungfun_core::MessageRole::System,
                        "Tool" => cheungfun_core::MessageRole::Tool,
                        _ => cheungfun_core::MessageRole::User, // Default fallback
                    };

                    let metadata = if metadata_value.is_null() {
                        None
                    } else {
                        serde_json::from_value(metadata_value).ok()
                    };

                    messages.push(ChatMessage {
                        role,
                        content,
                        timestamp,
                        metadata,
                    });
                }
            }
            DatabasePool::Sqlite(pool) => {
                let query = format!(
                    r#"
                    SELECT role, content, metadata, created_at
                    FROM {} 
                    WHERE conversation_key = ? 
                    ORDER BY message_order ASC
                    "#,
                    self.table_name
                );

                let rows = sqlx::query(&query)
                    .bind(key)
                    .fetch_all(pool)
                    .await
                    .map_err(|e| CheungfunError::VectorStore {
                        message: format!("Failed to fetch messages for {}: {}", key, e),
                    })?;

                for row in rows {
                    let role_str: String = row.get("role");
                    let content: String = row.get("content");
                    let metadata_str: String = row.get("metadata");
                    let timestamp_str: String = row.get("created_at");

                    let role = match role_str.as_str() {
                        "User" => cheungfun_core::MessageRole::User,
                        "Assistant" => cheungfun_core::MessageRole::Assistant,
                        "System" => cheungfun_core::MessageRole::System,
                        "Tool" => cheungfun_core::MessageRole::Tool,
                        _ => cheungfun_core::MessageRole::User, // Default fallback
                    };

                    let metadata = if metadata_str.is_empty() {
                        None
                    } else {
                        serde_json::from_str(&metadata_str).ok()
                    };

                    // Parse SQLite datetime string
                    let timestamp = chrono::DateTime::parse_from_rfc3339(&timestamp_str)
                        .unwrap_or_else(|_| chrono::Utc::now().into())
                        .with_timezone(&chrono::Utc);

                    messages.push(ChatMessage {
                        role,
                        content,
                        timestamp,
                        metadata,
                    });
                }
            }
        }

        Ok(messages)
    }

    async fn add_message(&self, key: &str, message: ChatMessage) -> Result<()> {
        // Get the next message order
        let next_order = self.count_messages(key).await?;

        match &self.pool {
            DatabasePool::Postgres(pool) => {
                let metadata_json = message
                    .metadata
                    .map(|m| serde_json::to_value(m).unwrap_or_default())
                    .unwrap_or_default();

                let query = format!(
                    r#"
                    INSERT INTO {} (conversation_key, role, content, metadata, message_order)
                    VALUES ($1, $2, $3, $4, $5)
                    "#,
                    self.table_name
                );

                sqlx::query(&query)
                    .bind(key)
                    .bind(format!("{:?}", message.role))
                    .bind(&message.content)
                    .bind(&metadata_json)
                    .bind(next_order as i32)
                    .execute(pool)
                    .await
                    .map_err(|e| CheungfunError::VectorStore {
                        message: format!("Failed to add message: {}", e),
                    })?;
            }
            DatabasePool::Sqlite(pool) => {
                let metadata_json = message
                    .metadata
                    .map(|m| serde_json::to_string(&m).unwrap_or_default())
                    .unwrap_or_default();

                let query = format!(
                    r#"
                    INSERT INTO {} (conversation_key, role, content, metadata, message_order)
                    VALUES (?, ?, ?, ?, ?)
                    "#,
                    self.table_name
                );

                sqlx::query(&query)
                    .bind(key)
                    .bind(format!("{:?}", message.role))
                    .bind(&message.content)
                    .bind(&metadata_json)
                    .bind(next_order as i32)
                    .execute(pool)
                    .await
                    .map_err(|e| CheungfunError::VectorStore {
                        message: format!("Failed to add message: {}", e),
                    })?;
            }
        }

        debug!("Added message to conversation: {}", key);
        Ok(())
    }

    async fn delete_messages(&self, key: &str) -> Result<()> {
        match &self.pool {
            DatabasePool::Postgres(pool) => {
                let query = format!(
                    "DELETE FROM {} WHERE conversation_key = $1",
                    self.table_name
                );
                sqlx::query(&query)
                    .bind(key)
                    .execute(pool)
                    .await
                    .map_err(|e| CheungfunError::VectorStore {
                        message: format!("Failed to delete messages for {}: {}", key, e),
                    })?;
            }
            DatabasePool::Sqlite(pool) => {
                let query = format!("DELETE FROM {} WHERE conversation_key = ?", self.table_name);
                sqlx::query(&query)
                    .bind(key)
                    .execute(pool)
                    .await
                    .map_err(|e| CheungfunError::VectorStore {
                        message: format!("Failed to delete messages for {}: {}", key, e),
                    })?;
            }
        }

        debug!("Deleted messages for conversation: {}", key);
        Ok(())
    }

    async fn get_keys(&self) -> Result<Vec<String>> {
        let mut keys = Vec::new();

        match &self.pool {
            DatabasePool::Postgres(pool) => {
                let query = format!("SELECT DISTINCT conversation_key FROM {}", self.table_name);
                let rows = sqlx::query(&query).fetch_all(pool).await.map_err(|e| {
                    CheungfunError::VectorStore {
                        message: format!("Failed to fetch conversation keys: {}", e),
                    }
                })?;

                for row in rows {
                    let key: String = row.get("conversation_key");
                    keys.push(key);
                }
            }
            DatabasePool::Sqlite(pool) => {
                let query = format!("SELECT DISTINCT conversation_key FROM {}", self.table_name);
                let rows = sqlx::query(&query).fetch_all(pool).await.map_err(|e| {
                    CheungfunError::VectorStore {
                        message: format!("Failed to fetch conversation keys: {}", e),
                    }
                })?;

                for row in rows {
                    let key: String = row.get("conversation_key");
                    keys.push(key);
                }
            }
        }

        Ok(keys)
    }

    async fn count_messages(&self, key: &str) -> Result<usize> {
        match &self.pool {
            DatabasePool::Postgres(pool) => {
                let query = format!(
                    "SELECT COUNT(*) as count FROM {} WHERE conversation_key = $1",
                    self.table_name
                );
                let row = sqlx::query(&query)
                    .bind(key)
                    .fetch_one(pool)
                    .await
                    .map_err(|e| CheungfunError::VectorStore {
                        message: format!("Failed to count messages for {}: {}", key, e),
                    })?;
                let count: i64 = row.get("count");
                Ok(count as usize)
            }
            DatabasePool::Sqlite(pool) => {
                let query = format!(
                    "SELECT COUNT(*) as count FROM {} WHERE conversation_key = ?",
                    self.table_name
                );
                let row = sqlx::query(&query)
                    .bind(key)
                    .fetch_one(pool)
                    .await
                    .map_err(|e| CheungfunError::VectorStore {
                        message: format!("Failed to count messages for {}: {}", key, e),
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
                        message: format!("Failed to clear conversations: {}", e),
                    }
                })?;
            }
            DatabasePool::Sqlite(pool) => {
                let query = format!("DELETE FROM {}", self.table_name);
                sqlx::query(&query).execute(pool).await.map_err(|e| {
                    CheungfunError::VectorStore {
                        message: format!("Failed to clear conversations: {}", e),
                    }
                })?;
            }
        }

        info!("Cleared all conversations from store");
        Ok(())
    }
}
