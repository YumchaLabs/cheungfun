# Cheungfun Multimodal

A comprehensive multimodal extension for the Cheungfun RAG framework, providing unified processing, embedding, and retrieval capabilities across text, image, audio, video, and document modalities.

## 🚀 Features

- **Unified Data Structures**: Common interfaces for all media types
- **Cross-Modal Embeddings**: Generate embeddings that work across modalities  
- **Cross-Modal Retrieval**: Search for images using text, audio using images, etc.
- **Format Support**: Wide range of media formats with automatic detection
- **Performance Optimized**: Batch processing, streaming, and GPU acceleration support
- **Extensible Architecture**: Plugin system for custom processors and embedders

## 📦 Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
cheungfun-multimodal = { path = "../cheungfun-multimodal", features = ["full"] }
```

### Feature Flags

- `image-support`: Enable image processing capabilities
- `audio-support`: Enable audio processing capabilities  
- `video-support`: Enable video processing capabilities
- `format-detection`: Enable automatic format detection
- `remote-resources`: Enable fetching from URLs
- `candle`: Enable Candle ML framework integration
- `clip`: Enable CLIP multimodal embeddings
- `full`: Enable all features

## 🎯 Quick Start

```rust
use cheungfun_multimodal::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize the library
    cheungfun_multimodal::init()?;
    
    // Create a multimodal document from an image
    let image_content = MediaContent::from_file_path("image.jpg", MediaFormat::Jpeg);
    let doc = MultimodalDocument::with_media(
        Document::new("A beautiful sunset"),
        ModalityType::Image,
        image_content,
    );

    // Create a cross-modal query
    let query = CrossModalQuery::from_text("Find images of sunsets");

    println!("Created multimodal document and query");
    Ok(())
}
```

## 🏗️ Architecture

The library is organized into several key modules:

- **`types`**: Core data structures for multimodal content
- **`traits`**: Trait definitions for processors, embedders, and retrievers
- **`processors`**: Implementations for different media processing tasks
- **`embedders`**: Multimodal embedding generation
- **`retrievers`**: Cross-modal search and retrieval
- **`utils`**: Utility functions and helpers

## 📊 Modality Support

| Modality | Read | Write | Embed | Search | Extract Text |
|----------|------|-------|-------|--------|--------------|
| Text     | ✅   | ✅    | ✅    | ✅     | ✅           |
| Image    | ✅   | ✅    | 🚧    | 🚧     | ❌           |
| Audio    | 🚧   | ❌    | 🚧    | 🚧     | 🚧*          |
| Video    | 🚧   | ❌    | 🚧    | 🚧     | 🚧*          |
| Document | 🚧   | ❌    | 🚧    | 🚧     | ✅           |

- ✅ Implemented
- 🚧 In Progress  
- ❌ Not Supported
- *Requires speech recognition features

## 🔧 Usage Examples

### Image Processing

```rust
use cheungfun_multimodal::prelude::*;

// Create image processor
let processor = ImageProcessor::builder()
    .max_dimensions(2048, 2048)
    .default_quality(0.9)
    .build();

// Load and process image
let content = MediaContent::from_file_path("image.jpg", MediaFormat::Jpeg);
let features = processor.extract_features(&content).await?;

println!("Image dimensions: {}x{}", 
         features["width"], features["height"]);
```

### Cross-Modal Query

```rust
use cheungfun_multimodal::prelude::*;

// Create a text query for images
let query = CrossModalQuery::from_text("sunset over mountains")
    .with_metadata("category", "landscape");

// Set up retrieval options
let options = CrossModalRetrievalOptions {
    top_k: 10,
    threshold: Some(0.7),
    include_explanations: true,
    ..Default::default()
};
```

### Multimodal Document

```rust
use cheungfun_multimodal::prelude::*;

// Create a document with multiple modalities
let mut doc = MultimodalDocument::from_document(
    Document::new("Product description")
);

// Add product image
let image = MediaContent::from_file_path("product.jpg", MediaFormat::Jpeg);
doc = doc.add_related_media(image);

// Add product video
let video = MediaContent::from_file_path("demo.mp4", MediaFormat::Mp4);
doc = doc.add_related_media(video);

println!("Document has {} modalities", doc.get_modalities().len());
```

## 🛠️ Development Status

This is an early-stage implementation focusing on:

1. ✅ **Core Architecture**: Data structures and trait definitions
2. ✅ **Image Processing**: Basic image handling and feature extraction
3. 🚧 **Audio Processing**: Framework in place, implementation pending
4. 🚧 **Video Processing**: Framework in place, implementation pending  
5. 🚧 **Multimodal Embeddings**: Interface defined, implementations pending
6. 🚧 **Cross-Modal Retrieval**: Interface defined, implementations pending

## 🔮 Roadmap

### Phase 1: Foundation (Current)
- [x] Core data structures and traits
- [x] Basic image processing
- [x] Format detection utilities
- [x] Documentation and examples

### Phase 2: Core Implementations
- [ ] CLIP multimodal embeddings
- [ ] Audio processing with speech recognition
- [ ] Video frame extraction and analysis
- [ ] Cross-modal retrieval system

### Phase 3: Advanced Features
- [ ] GPU acceleration support
- [ ] Streaming processing
- [ ] Advanced ML models integration
- [ ] Performance optimizations

### Phase 4: Production Ready
- [ ] Comprehensive testing
- [ ] Benchmarking and optimization
- [ ] Production deployment guides
- [ ] Integration with existing RAG systems

## 🤝 Contributing

This project is part of the Cheungfun RAG framework. Contributions are welcome!

1. Check the current task list and roadmap
2. Pick an area that interests you
3. Follow the existing code patterns and architecture
4. Add tests for new functionality
5. Update documentation

## 📄 License

This project is licensed under the MIT OR Apache-2.0 license.

## 🙏 Acknowledgments

- Built on top of the Cheungfun RAG framework
- Inspired by LlamaIndex's multimodal capabilities
- Uses the Candle ML framework for deep learning
- Leverages the Rust ecosystem for performance and safety
