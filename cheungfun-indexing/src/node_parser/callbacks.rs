//! Callback and event system for node parsers.
//!
//! This module provides a comprehensive callback system that allows monitoring
//! and instrumentation of the node parsing process. It follows the LlamaIndex
//! callback pattern while leveraging Rust's async capabilities.

use crate::loaders::ProgrammingLanguage;
use async_trait::async_trait;
use cheungfun_core::{Node, Result as CoreResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Callback event types that can occur during node parsing.
///
/// These events correspond to different stages of the parsing process
/// and allow for detailed monitoring and instrumentation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CallbackEventType {
    /// Node parsing process started.
    NodeParsingStart,
    /// Node parsing process completed.
    NodeParsingEnd,
    /// Text chunking process started.
    ChunkingStart,
    /// Text chunking process completed.
    ChunkingEnd,
    /// AST parsing process started.
    AstParsingStart,
    /// AST parsing process completed.
    AstParsingEnd,
    /// Semantic analysis started.
    SemanticAnalysisStart,
    /// Semantic analysis completed.
    SemanticAnalysisEnd,
    /// Error occurred during processing.
    Error,
    /// Custom event type.
    Custom(String),
}

/// Event payload containing data associated with a callback event.
///
/// This structure holds the actual data that accompanies each event,
/// allowing handlers to access relevant information about the parsing process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventPayload {
    /// The type of event that occurred.
    pub event_type: CallbackEventType,
    /// Event-specific data.
    pub data: HashMap<String, serde_json::Value>,
    /// Timestamp when the event occurred.
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Unique identifier for this event.
    pub event_id: Uuid,
    /// Duration of the operation (for end events).
    pub duration: Option<Duration>,
}

impl EventPayload {
    /// Create a new event payload.
    pub fn new(event_type: CallbackEventType) -> Self {
        Self {
            event_type,
            data: HashMap::new(),
            timestamp: chrono::Utc::now(),
            event_id: Uuid::new_v4(),
            duration: None,
        }
    }

    /// Add data to the event payload.
    pub fn with_data<K: Into<String>>(mut self, key: K, value: serde_json::Value) -> Self {
        self.data.insert(key.into(), value);
        self
    }

    /// Set the duration for this event.
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Get data value by key.
    pub fn get_data(&self, key: &str) -> Option<&serde_json::Value> {
        self.data.get(key)
    }
}

/// Trait for callback handlers.
///
/// Implementations of this trait can be registered with the callback manager
/// to receive notifications about parsing events.
#[async_trait]
pub trait CallbackHandler: Send + Sync {
    /// Handle a callback event.
    ///
    /// # Arguments
    ///
    /// * `payload` - The event payload containing event data
    ///
    /// # Returns
    ///
    /// Result indicating success or failure of event handling.
    async fn on_event(&self, payload: &EventPayload) -> CoreResult<()>;

    /// Get the name of this callback handler.
    fn name(&self) -> &str;

    /// Check if this handler is interested in a specific event type.
    fn handles_event_type(&self, event_type: &CallbackEventType) -> bool {
        // By default, handle all event types
        true
    }
}

/// Logging callback handler that outputs events to the logging system.
#[derive(Debug, Clone)]
pub struct LoggingCallbackHandler {
    name: String,
    log_level: tracing::Level,
}

impl LoggingCallbackHandler {
    /// Create a new logging callback handler.
    pub fn new<S: Into<String>>(name: S) -> Self {
        Self {
            name: name.into(),
            log_level: tracing::Level::INFO,
        }
    }

    /// Set the log level for this handler.
    pub fn with_log_level(mut self, level: tracing::Level) -> Self {
        self.log_level = level;
        self
    }
}

#[async_trait]
impl CallbackHandler for LoggingCallbackHandler {
    async fn on_event(&self, payload: &EventPayload) -> CoreResult<()> {
        let message = format!(
            "[{}] Event: {:?} at {} (ID: {})",
            self.name,
            payload.event_type,
            payload.timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
            payload.event_id
        );

        match self.log_level {
            tracing::Level::ERROR => error!("{}", message),
            tracing::Level::WARN => warn!("{}", message),
            tracing::Level::INFO => info!("{}", message),
            tracing::Level::DEBUG => debug!("{}", message),
            tracing::Level::TRACE => tracing::trace!("{}", message),
        }

        // Log additional data if present
        if !payload.data.is_empty() {
            debug!("Event data: {:?}", payload.data);
        }

        if let Some(duration) = payload.duration {
            debug!("Event duration: {:?}", duration);
        }

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Metrics callback handler that collects performance metrics.
#[derive(Debug, Clone)]
pub struct MetricsCallbackHandler {
    name: String,
    metrics: Arc<std::sync::Mutex<HashMap<String, f64>>>,
}

impl MetricsCallbackHandler {
    /// Create a new metrics callback handler.
    pub fn new<S: Into<String>>(name: S) -> Self {
        Self {
            name: name.into(),
            metrics: Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }

    /// Get collected metrics.
    pub fn get_metrics(&self) -> HashMap<String, f64> {
        self.metrics.lock().unwrap().clone()
    }

    /// Reset collected metrics.
    pub fn reset_metrics(&self) {
        self.metrics.lock().unwrap().clear();
    }
}

#[async_trait]
impl CallbackHandler for MetricsCallbackHandler {
    async fn on_event(&self, payload: &EventPayload) -> CoreResult<()> {
        let mut metrics = self.metrics.lock().unwrap();

        // Count events
        let event_key = format!("{:?}_count", payload.event_type);
        *metrics.entry(event_key).or_insert(0.0) += 1.0;

        // Record durations
        if let Some(duration) = payload.duration {
            let duration_key = format!("{:?}_duration_ms", payload.event_type);
            metrics.insert(duration_key, duration.as_millis() as f64);
        }

        // Record specific metrics based on event data
        match payload.event_type {
            CallbackEventType::ChunkingEnd => {
                if let Some(chunks) = payload.get_data("chunks") {
                    if let Some(chunks_array) = chunks.as_array() {
                        metrics.insert("chunks_created".to_string(), chunks_array.len() as f64);
                    }
                }
            }
            CallbackEventType::NodeParsingEnd => {
                if let Some(nodes) = payload.get_data("nodes") {
                    if let Some(nodes_array) = nodes.as_array() {
                        metrics.insert("nodes_created".to_string(), nodes_array.len() as f64);
                    }
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Callback manager that coordinates multiple callback handlers.
///
/// This manager maintains a list of callback handlers and dispatches
/// events to all registered handlers.
#[derive(Clone)]
pub struct CallbackManager {
    handlers: Vec<Arc<dyn CallbackHandler>>,
}

impl std::fmt::Debug for CallbackManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CallbackManager")
            .field("handlers", &format!("{} handlers", self.handlers.len()))
            .finish()
    }
}

impl CallbackManager {
    /// Create a new callback manager.
    pub fn new() -> Self {
        Self {
            handlers: Vec::new(),
        }
    }

    /// Create a callback manager with default handlers.
    pub fn with_defaults() -> Self {
        let mut manager = Self::new();
        manager.add_handler(Arc::new(LoggingCallbackHandler::new("default")));
        manager
    }

    /// Add a callback handler.
    pub fn add_handler(&mut self, handler: Arc<dyn CallbackHandler>) {
        self.handlers.push(handler);
    }

    /// Remove a callback handler by name.
    pub fn remove_handler(&mut self, name: &str) {
        self.handlers.retain(|h| h.name() != name);
    }

    /// Emit an event to all registered handlers.
    pub async fn emit_event(&self, payload: EventPayload) -> CoreResult<()> {
        for handler in &self.handlers {
            if handler.handles_event_type(&payload.event_type) {
                if let Err(e) = handler.on_event(&payload).await {
                    error!("Callback handler '{}' failed: {}", handler.name(), e);
                    // Continue with other handlers even if one fails
                }
            }
        }
        Ok(())
    }

    /// Create an event context for tracking start/end events.
    pub fn event_context(&self, event_type: CallbackEventType) -> EventContext {
        EventContext::new(self.clone(), event_type)
    }
}

impl Default for CallbackManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Event context for tracking start/end event pairs.
///
/// This structure automatically emits start events when created and
/// end events when dropped, making it easy to track operation durations.
pub struct EventContext {
    manager: CallbackManager,
    event_type: CallbackEventType,
    start_time: Instant,
    start_payload: EventPayload,
}

impl EventContext {
    /// Create a new event context.
    fn new(manager: CallbackManager, event_type: CallbackEventType) -> Self {
        let start_time = Instant::now();
        let start_payload = EventPayload::new(event_type.clone());

        Self {
            manager,
            event_type,
            start_time,
            start_payload,
        }
    }

    /// Emit the start event.
    pub async fn start(mut self) -> Self {
        if let Err(e) = self.manager.emit_event(self.start_payload.clone()).await {
            error!("Failed to emit start event: {}", e);
        }
        self
    }

    /// Add data to the event context.
    pub fn with_data<K: Into<String>>(mut self, key: K, value: serde_json::Value) -> Self {
        self.start_payload = self.start_payload.with_data(key, value);
        self
    }

    /// Emit the end event with duration.
    pub async fn end(self, end_data: HashMap<String, serde_json::Value>) {
        let duration = self.start_time.elapsed();

        let end_event_type = match self.event_type {
            CallbackEventType::NodeParsingStart => CallbackEventType::NodeParsingEnd,
            CallbackEventType::ChunkingStart => CallbackEventType::ChunkingEnd,
            CallbackEventType::AstParsingStart => CallbackEventType::AstParsingEnd,
            CallbackEventType::SemanticAnalysisStart => CallbackEventType::SemanticAnalysisEnd,
            _ => self.event_type,
        };

        let mut end_payload = EventPayload::new(end_event_type).with_duration(duration);
        for (key, value) in end_data {
            end_payload = end_payload.with_data(key, value);
        }

        if let Err(e) = self.manager.emit_event(end_payload).await {
            error!("Failed to emit end event: {}", e);
        }
    }
}

/// Convenience functions for creating common event payloads.
impl EventPayload {
    /// Create a chunking start event.
    pub fn chunking_start(text_length: usize) -> Self {
        Self::new(CallbackEventType::ChunkingStart)
            .with_data("text_length", serde_json::Value::Number(text_length.into()))
    }

    /// Create a chunking end event.
    pub fn chunking_end(chunks: &[String]) -> Self {
        let chunks_json: Vec<serde_json::Value> = chunks
            .iter()
            .map(|s| serde_json::Value::String(s.clone()))
            .collect();

        Self::new(CallbackEventType::ChunkingEnd)
            .with_data("chunks", serde_json::Value::Array(chunks_json))
            .with_data(
                "chunk_count",
                serde_json::Value::Number(chunks.len().into()),
            )
    }

    /// Create a node parsing start event.
    pub fn node_parsing_start(document_count: usize) -> Self {
        Self::new(CallbackEventType::NodeParsingStart).with_data(
            "document_count",
            serde_json::Value::Number(document_count.into()),
        )
    }

    /// Create a node parsing end event.
    pub fn node_parsing_end(nodes: &[Node]) -> Self {
        Self::new(CallbackEventType::NodeParsingEnd)
            .with_data("node_count", serde_json::Value::Number(nodes.len().into()))
    }

    /// Create an AST parsing start event.
    pub fn ast_parsing_start(language: ProgrammingLanguage) -> Self {
        Self::new(CallbackEventType::AstParsingStart).with_data(
            "language",
            serde_json::Value::String(language.as_str().to_string()),
        )
    }

    /// Create an AST parsing end event.
    pub fn ast_parsing_end(functions: usize, classes: usize) -> Self {
        Self::new(CallbackEventType::AstParsingEnd)
            .with_data("functions", serde_json::Value::Number(functions.into()))
            .with_data("classes", serde_json::Value::Number(classes.into()))
    }
}
