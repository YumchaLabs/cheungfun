-- Initial migration for SQLite
-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    content_hash TEXT,
    embedding BLOB,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create chat_messages table
CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    conversation_key TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    metadata TEXT DEFAULT '{}',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create index_structs table
CREATE TABLE IF NOT EXISTS index_structs (
    index_id TEXT PRIMARY KEY,
    index_type TEXT NOT NULL,
    config TEXT NOT NULL DEFAULT '{}',
    node_ids TEXT NOT NULL DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_key ON chat_messages(conversation_key);
CREATE INDEX IF NOT EXISTS idx_chat_messages_timestamp ON chat_messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
CREATE INDEX IF NOT EXISTS idx_index_structs_type ON index_structs(index_type);
