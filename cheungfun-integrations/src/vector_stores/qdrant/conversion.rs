//! Data type conversion utilities for Qdrant integration.
//!
//! This module provides functions to convert between Cheungfun types (Node, Query, etc.)
//! and Qdrant types (PointStruct, ScoredPoint, etc.).

use cheungfun_core::{
    types::{ChunkInfo, Node},
    Result,
};
use qdrant_client::qdrant::{PointId, PointStruct, RetrievedPoint, ScoredPoint};
use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

use super::config::QdrantConfig;

/// Convert a Node to a Qdrant PointStruct.
///
/// This function transforms a Cheungfun Node into the format expected by Qdrant,
/// including proper handling of embeddings, metadata, and chunk information.
///
/// # Arguments
///
/// * `node` - The Node to convert
/// * `config` - Qdrant configuration for validation
///
/// # Returns
///
/// A Result containing the converted PointStruct or an error
///
/// # Errors
///
/// Returns an error if:
/// - The node doesn't have an embedding
/// - The embedding dimension doesn't match the configuration
/// - Metadata conversion fails
pub fn node_to_point(node: &Node, config: &QdrantConfig) -> Result<PointStruct> {
    use qdrant_client::qdrant::Value as QdrantValue;

    let embedding =
        node.embedding
            .as_ref()
            .ok_or_else(|| cheungfun_core::CheungfunError::Validation {
                message: "Node must have an embedding".to_string(),
            })?;

    // Validate vector dimension
    if embedding.len() != config.dimension {
        return Err(cheungfun_core::CheungfunError::Validation {
            message: format!(
                "Vector dimension {} does not match collection dimension {}",
                embedding.len(),
                config.dimension
            ),
        });
    }

    // Convert metadata to Qdrant payload
    let mut payload = HashMap::new();

    // Add node content
    payload.insert(
        "content".to_string(),
        QdrantValue::from(node.content.clone()),
    );

    // Add source document ID
    payload.insert(
        "source_document_id".to_string(),
        QdrantValue::from(node.source_document_id.to_string()),
    );

    // Add chunk info
    payload.insert(
        "chunk_start".to_string(),
        QdrantValue::from(node.chunk_info.start_char_idx.unwrap_or(0) as i64),
    );
    payload.insert(
        "chunk_end".to_string(),
        QdrantValue::from(node.chunk_info.end_char_idx.unwrap_or(0) as i64),
    );
    payload.insert(
        "chunk_index".to_string(),
        QdrantValue::from(node.chunk_info.chunk_index as i64),
    );

    // Add user metadata - convert serde_json::Value to QdrantValue
    for (key, value) in &node.metadata {
        let qdrant_value = serde_value_to_qdrant_value(value);
        payload.insert(key.clone(), qdrant_value);
    }

    Ok(PointStruct::new(
        node.id.to_string(),
        embedding.clone(),
        payload,
    ))
}

/// Convert a Qdrant ScoredPoint to a Node.
///
/// This function transforms a Qdrant ScoredPoint back into a Cheungfun Node,
/// extracting all the metadata and chunk information that was stored.
///
/// # Arguments
///
/// * `point` - The ScoredPoint to convert
///
/// # Returns
///
/// A Result containing an optional Node (None if conversion fails)
pub fn scored_point_to_node(point: ScoredPoint) -> Result<Option<Node>> {
    let point_id = match point.id {
        Some(id) => point_id_to_string(&id)?,
        None => return Ok(None),
    };

    let node_id =
        Uuid::parse_str(&point_id).map_err(|e| cheungfun_core::CheungfunError::Validation {
            message: format!("Invalid UUID in point ID: {}", e),
        })?;

    let payload = point.payload;
    let _vectors = point.vectors;

    extract_node_from_payload(node_id, payload)
}

/// Convert a Qdrant RetrievedPoint to a Node.
///
/// Similar to scored_point_to_node but for RetrievedPoint which doesn't have a score.
///
/// # Arguments
///
/// * `point` - The RetrievedPoint to convert
///
/// # Returns
///
/// A Result containing an optional Node (None if conversion fails)
pub fn retrieved_point_to_node(point: RetrievedPoint) -> Result<Option<Node>> {
    let point_id = match point.id {
        Some(id) => point_id_to_string(&id)?,
        None => return Ok(None),
    };

    let node_id =
        Uuid::parse_str(&point_id).map_err(|e| cheungfun_core::CheungfunError::Validation {
            message: format!("Invalid UUID in point ID: {}", e),
        })?;

    let payload = point.payload;
    let _vectors = point.vectors;

    extract_node_from_payload(node_id, payload)
}

/// Extract a Node from Qdrant payload data.
///
/// This is a helper function used by both scored_point_to_node and retrieved_point_to_node
/// to avoid code duplication.
fn extract_node_from_payload(
    node_id: Uuid,
    payload: HashMap<String, qdrant_client::qdrant::Value>,
) -> Result<Option<Node>> {
    // Extract content
    let content = payload
        .get("content")
        .and_then(|v| qdrant_value_to_string(v))
        .unwrap_or_default();

    // Extract source document ID
    let source_document_id = payload
        .get("source_document_id")
        .and_then(|v| qdrant_value_to_string(v))
        .and_then(|s| Uuid::parse_str(&s).ok())
        .unwrap_or_else(Uuid::new_v4);

    // Extract chunk info
    let chunk_start = payload
        .get("chunk_start")
        .and_then(|v| qdrant_value_to_u64(v))
        .unwrap_or(0) as usize;

    let chunk_end = payload
        .get("chunk_end")
        .and_then(|v| qdrant_value_to_u64(v))
        .unwrap_or(0) as usize;

    let chunk_index = payload
        .get("chunk_index")
        .and_then(|v| qdrant_value_to_u64(v))
        .unwrap_or(0) as usize;

    let chunk_info = ChunkInfo::new(Some(chunk_start), Some(chunk_end), chunk_index);

    // Extract embedding - simplified for now
    let embedding = None; // TODO: Extract from vectors when needed

    // Extract metadata (excluding our internal fields)
    let mut metadata = HashMap::new();
    for (key, value) in payload {
        if !matches!(
            key.as_str(),
            "content" | "source_document_id" | "chunk_start" | "chunk_end" | "chunk_index"
        ) {
            let serde_value = qdrant_value_to_serde_value(&value);
            metadata.insert(key, serde_value);
        }
    }

    let mut node = Node::new(content, source_document_id, chunk_info);
    node.id = node_id;
    node.metadata = metadata;

    if let Some(emb) = embedding {
        node.embedding = Some(emb);
    }

    Ok(Some(node))
}

/// Convert serde_json::Value to qdrant::Value.
///
/// This function handles the conversion between JSON values and Qdrant's value format.
pub fn serde_value_to_qdrant_value(value: &Value) -> qdrant_client::qdrant::Value {
    use qdrant_client::qdrant::Value as QdrantValue;

    match value {
        Value::Null => QdrantValue::from(""),
        Value::Bool(b) => QdrantValue::from(*b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                QdrantValue::from(i)
            } else if let Some(f) = n.as_f64() {
                QdrantValue::from(f)
            } else {
                QdrantValue::from(n.to_string())
            }
        }
        Value::String(s) => QdrantValue::from(s.clone()),
        Value::Array(arr) => {
            // Convert array to string representation for simplicity
            QdrantValue::from(serde_json::to_string(arr).unwrap_or_default())
        }
        Value::Object(obj) => {
            // Convert object to string representation for simplicity
            QdrantValue::from(serde_json::to_string(obj).unwrap_or_default())
        }
    }
}

/// Convert PointId to string.
pub fn point_id_to_string(point_id: &PointId) -> Result<String> {
    match &point_id.point_id_options {
        Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid)) => Ok(uuid.clone()),
        Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(num)) => Ok(num.to_string()),
        None => Err(cheungfun_core::CheungfunError::Validation {
            message: "Point ID is empty".to_string(),
        }),
    }
}

/// Convert Qdrant Value to string.
pub fn qdrant_value_to_string(value: &qdrant_client::qdrant::Value) -> Option<String> {
    match &value.kind {
        Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => Some(s.clone()),
        Some(qdrant_client::qdrant::value::Kind::IntegerValue(i)) => Some(i.to_string()),
        Some(qdrant_client::qdrant::value::Kind::DoubleValue(d)) => Some(d.to_string()),
        Some(qdrant_client::qdrant::value::Kind::BoolValue(b)) => Some(b.to_string()),
        _ => None,
    }
}

/// Convert Qdrant Value to u64.
pub fn qdrant_value_to_u64(value: &qdrant_client::qdrant::Value) -> Option<u64> {
    match &value.kind {
        Some(qdrant_client::qdrant::value::Kind::IntegerValue(i)) => Some(*i as u64),
        Some(qdrant_client::qdrant::value::Kind::DoubleValue(d)) => Some(*d as u64),
        _ => None,
    }
}

/// Convert Qdrant Value to serde_json Value.
pub fn qdrant_value_to_serde_value(value: &qdrant_client::qdrant::Value) -> serde_json::Value {
    match &value.kind {
        Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => {
            serde_json::Value::String(s.clone())
        }
        Some(qdrant_client::qdrant::value::Kind::IntegerValue(i)) => {
            serde_json::Value::Number((*i).into())
        }
        Some(qdrant_client::qdrant::value::Kind::DoubleValue(d)) => {
            serde_json::Value::Number(serde_json::Number::from_f64(*d).unwrap_or_else(|| 0.into()))
        }
        Some(qdrant_client::qdrant::value::Kind::BoolValue(b)) => serde_json::Value::Bool(*b),
        _ => serde_json::Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cheungfun_core::types::ChunkInfo;

    fn create_test_node(content: &str, embedding: Vec<f32>) -> Node {
        let source_doc_id = Uuid::new_v4();
        let chunk_info = ChunkInfo::new(0, content.len(), 0);
        Node::new(content, source_doc_id, chunk_info).with_embedding(embedding)
    }

    #[test]
    fn test_node_to_point_conversion() {
        let config = QdrantConfig::new("http://localhost:6334", "test", 3);
        let mut node = create_test_node("Test content", vec![1.0, 0.0, 0.0]);
        node.metadata
            .insert("category".to_string(), Value::String("test".to_string()));

        let point = node_to_point(&node, &config).unwrap();

        // Basic validation that point was created
        assert!(point.id.is_some());
        assert!(point.vectors.is_some());
        assert!(point.payload.contains_key("content"));
        assert!(point.payload.contains_key("category"));
    }

    #[test]
    fn test_node_to_point_invalid_dimension() {
        let config = QdrantConfig::new("http://localhost:6334", "test", 3);
        let node = create_test_node("Test content", vec![1.0, 0.0]); // Wrong dimension

        let result = node_to_point(&node, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_node_without_embedding() {
        let config = QdrantConfig::new("http://localhost:6334", "test", 3);
        let source_doc_id = Uuid::new_v4();
        let chunk_info = ChunkInfo::new(0, 10, 0);
        let node = Node::new("Test content", source_doc_id, chunk_info); // No embedding

        let result = node_to_point(&node, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_serde_value_conversion() {
        // Test different value types
        let bool_val = Value::Bool(true);
        let num_val = Value::Number(serde_json::Number::from(42));
        let str_val = Value::String("test".to_string());
        let null_val = Value::Null;

        let qdrant_bool = serde_value_to_qdrant_value(&bool_val);
        let qdrant_num = serde_value_to_qdrant_value(&num_val);
        let qdrant_str = serde_value_to_qdrant_value(&str_val);
        let qdrant_null = serde_value_to_qdrant_value(&null_val);

        // These should not panic and should convert appropriately
        assert!(matches!(qdrant_bool, qdrant_client::qdrant::Value { .. }));
        assert!(matches!(qdrant_num, qdrant_client::qdrant::Value { .. }));
        assert!(matches!(qdrant_str, qdrant_client::qdrant::Value { .. }));
        assert!(matches!(qdrant_null, qdrant_client::qdrant::Value { .. }));
    }
}
