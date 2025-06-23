//! Tokenizer integration for Candle embedder.
//!
//! This module provides text tokenization capabilities using HuggingFace tokenizers.
//! It handles text preprocessing, tokenization, and preparation of input tensors
//! for the embedding model.

use super::error::CandleError;
use candle_core::{Device, Tensor};
use std::path::Path;
use tokenizers::{Encoding, Tokenizer};
use tracing::debug;

/// Tokenizer wrapper for embedding models.
#[derive(Debug)]
pub struct EmbeddingTokenizer {
    tokenizer: Tokenizer,
    max_length: usize,
    pad_token_id: u32,
    cls_token_id: Option<u32>,
    sep_token_id: Option<u32>,
}

impl EmbeddingTokenizer {
    /// Create a new tokenizer from a file path.
    pub fn from_file<P: AsRef<Path>>(
        tokenizer_path: P,
        max_length: usize,
    ) -> Result<Self, CandleError> {
        let tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|e| CandleError::Tokenization {
                message: format!("Failed to load tokenizer: {}", e),
            })?;

        Self::from_tokenizer(tokenizer, max_length)
    }

    /// Create a new tokenizer from a Tokenizer instance.
    pub fn from_tokenizer(tokenizer: Tokenizer, max_length: usize) -> Result<Self, CandleError> {
        // Get special token IDs
        let vocab = tokenizer.get_vocab(true);
        let pad_token_id = vocab
            .get("[PAD]")
            .or_else(|| vocab.get("<pad>"))
            .or_else(|| vocab.get("</s>"))
            .copied()
            .unwrap_or(0);

        let cls_token_id = vocab.get("[CLS]").or_else(|| vocab.get("<s>")).copied();

        let sep_token_id = vocab.get("[SEP]").or_else(|| vocab.get("</s>")).copied();

        debug!(
            "Tokenizer initialized: pad_token_id={}, cls_token_id={:?}, sep_token_id={:?}",
            pad_token_id, cls_token_id, sep_token_id
        );

        Ok(Self {
            tokenizer,
            max_length,
            pad_token_id,
            cls_token_id,
            sep_token_id,
        })
    }

    /// Tokenize a single text.
    pub fn tokenize(&self, text: &str) -> Result<TokenizedInput, CandleError> {
        let encoding =
            self.tokenizer
                .encode(text, true)
                .map_err(|e| CandleError::Tokenization {
                    message: format!("Failed to encode text: {}", e),
                })?;

        self.process_encoding(encoding)
    }

    /// Tokenize multiple texts in batch.
    pub fn tokenize_batch(&self, texts: Vec<&str>) -> Result<Vec<TokenizedInput>, CandleError> {
        let encodings =
            self.tokenizer
                .encode_batch(texts, true)
                .map_err(|e| CandleError::Tokenization {
                    message: format!("Failed to encode batch: {}", e),
                })?;

        encodings
            .into_iter()
            .map(|encoding| self.process_encoding(encoding))
            .collect()
    }

    /// Process a single encoding into tokenized input.
    fn process_encoding(&self, mut encoding: Encoding) -> Result<TokenizedInput, CandleError> {
        // Truncate if necessary
        if encoding.len() > self.max_length {
            encoding.truncate(self.max_length, 0, tokenizers::TruncationDirection::Right);
            debug!("Truncated input to {} tokens", self.max_length);
        }

        let input_ids = encoding.get_ids().to_vec();
        let attention_mask = encoding.get_attention_mask().to_vec();
        let token_type_ids = encoding.get_type_ids().to_vec();

        // Pad to max_length if necessary
        let padded_input_ids = self.pad_sequence(&input_ids, self.pad_token_id);
        let padded_attention_mask = self.pad_sequence(&attention_mask, 0);
        let padded_token_type_ids = self.pad_sequence(&token_type_ids, 0);

        Ok(TokenizedInput {
            input_ids: padded_input_ids,
            attention_mask: padded_attention_mask,
            token_type_ids: padded_token_type_ids,
            original_length: input_ids.len(),
        })
    }

    /// Pad a sequence to max_length.
    fn pad_sequence(&self, sequence: &[u32], pad_value: u32) -> Vec<u32> {
        let mut padded = sequence.to_vec();
        while padded.len() < self.max_length {
            padded.push(pad_value);
        }
        padded
    }

    /// Convert tokenized inputs to tensors.
    pub fn to_tensors(
        &self,
        inputs: &[TokenizedInput],
        device: &Device,
    ) -> Result<ModelInputs, CandleError> {
        if inputs.is_empty() {
            return Err(CandleError::Tokenization {
                message: "No inputs provided".to_string(),
            });
        }

        let batch_size = inputs.len();
        let seq_len = self.max_length;

        // Collect all input IDs, attention masks, and token type IDs
        let mut all_input_ids = Vec::with_capacity(batch_size * seq_len);
        let mut all_attention_mask = Vec::with_capacity(batch_size * seq_len);
        let mut all_token_type_ids = Vec::with_capacity(batch_size * seq_len);

        for input in inputs {
            all_input_ids.extend_from_slice(&input.input_ids);
            all_attention_mask.extend_from_slice(&input.attention_mask);
            all_token_type_ids.extend_from_slice(&input.token_type_ids);
        }

        // Convert to tensors
        let input_ids =
            Tensor::from_vec(all_input_ids, (batch_size, seq_len), device).map_err(|e| {
                CandleError::Inference {
                    message: format!("Failed to create input_ids tensor: {}", e),
                }
            })?;

        let attention_mask = Tensor::from_vec(all_attention_mask, (batch_size, seq_len), device)
            .map_err(|e| CandleError::Inference {
                message: format!("Failed to create attention_mask tensor: {}", e),
            })?;

        let token_type_ids = Tensor::from_vec(all_token_type_ids, (batch_size, seq_len), device)
            .map_err(|e| CandleError::Inference {
                message: format!("Failed to create token_type_ids tensor: {}", e),
            })?;

        Ok(ModelInputs {
            input_ids,
            attention_mask,
            token_type_ids,
        })
    }

    /// Get the maximum sequence length.
    pub fn max_length(&self) -> usize {
        self.max_length
    }

    /// Get the pad token ID.
    pub fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
}

/// Tokenized input for a single text.
#[derive(Debug, Clone)]
pub struct TokenizedInput {
    /// Token IDs
    pub input_ids: Vec<u32>,
    /// Attention mask (1 for real tokens, 0 for padding)
    pub attention_mask: Vec<u32>,
    /// Token type IDs (for models that use them)
    pub token_type_ids: Vec<u32>,
    /// Original sequence length before padding
    pub original_length: usize,
}

/// Model inputs as tensors.
#[derive(Debug)]
pub struct ModelInputs {
    /// Input token IDs tensor [batch_size, seq_len]
    pub input_ids: Tensor,
    /// Attention mask tensor [batch_size, seq_len]
    pub attention_mask: Tensor,
    /// Token type IDs tensor [batch_size, seq_len]
    pub token_type_ids: Tensor,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    // Note: These tests require a real tokenizer file to run
    // In practice, you would load a tokenizer from HuggingFace Hub

    #[test]
    fn test_tokenized_input_creation() {
        let input = TokenizedInput {
            input_ids: vec![101, 7592, 2088, 102],
            attention_mask: vec![1, 1, 1, 1],
            token_type_ids: vec![0, 0, 0, 0],
            original_length: 4,
        };

        assert_eq!(input.input_ids.len(), 4);
        assert_eq!(input.attention_mask.len(), 4);
        assert_eq!(input.token_type_ids.len(), 4);
        assert_eq!(input.original_length, 4);
    }
}
