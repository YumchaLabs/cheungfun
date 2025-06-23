//! Image processing example.
//!
//! This example demonstrates how to use the multimodal framework to process
//! images, extract features, and perform basic image operations.

use cheungfun_multimodal::prelude::*;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize the multimodal library
    cheungfun_multimodal::init()?;
    
    println!("🖼️  Cheungfun Multimodal - Image Processing Example");
    println!("================================================");
    
    // Example 1: Create image content from bytes
    println!("\n📝 Example 1: Creating image content from bytes");
    let image_data = create_sample_image_data();
    let image_content = MediaContent::from_bytes(image_data, MediaFormat::Png)
        .with_metadata(
            MediaMetadata::new()
                .with_title("Sample Image")
                .with_description("A programmatically generated sample image")
                .with_tags(vec!["sample", "generated", "test"])
        );
    
    println!("✅ Created image content:");
    println!("   - Format: {}", image_content.format);
    println!("   - Modality: {}", image_content.modality_type());
    println!("   - Size: {} bytes", image_content.estimated_size().unwrap_or(0));
    println!("   - Title: {}", image_content.metadata.title.as_deref().unwrap_or("N/A"));
    
    // Example 2: Create image processor and extract features
    #[cfg(feature = "image-support")]
    {
        println!("\n🔍 Example 2: Extracting image features");
        let processor = cheungfun_multimodal::processors::ImageProcessor::builder()
            .max_dimensions(2048, 2048)
            .default_quality(0.9)
            .build();
        
        match processor.extract_features(&image_content).await {
            Ok(features) => {
                println!("✅ Extracted features:");
                for (key, value) in &features {
                    println!("   - {}: {}", key, value);
                }
            }
            Err(e) => {
                println!("❌ Failed to extract features: {}", e);
            }
        }
        
        // Example 3: Process image with different options
        println!("\n⚙️  Example 3: Processing image with options");
        let processing_options = ProcessingOptions {
            target_format: Some(MediaFormat::Jpeg),
            quality: Some(0.8),
            preserve_metadata: true,
            ..Default::default()
        };
        
        match processor.process(&image_content, &processing_options).await {
            Ok(processed_content) => {
                println!("✅ Processed image:");
                println!("   - Original format: {}", image_content.format);
                println!("   - New format: {}", processed_content.format);
                println!("   - Original size: {} bytes", image_content.estimated_size().unwrap_or(0));
                println!("   - New size: {} bytes", processed_content.estimated_size().unwrap_or(0));
            }
            Err(e) => {
                println!("❌ Failed to process image: {}", e);
            }
        }
    }
    
    #[cfg(not(feature = "image-support"))]
    {
        println!("\n⚠️  Image processing features are not enabled.");
        println!("   Enable with: --features image-support");
    }
    
    // Example 4: Create multimodal document with image
    println!("\n📄 Example 4: Creating multimodal document");
    let base_document = Document::new("This is a sample image document");
    let multimodal_doc = MultimodalDocument::with_media(
        base_document,
        ModalityType::Image,
        image_content.clone(),
    )
    .with_multimodal_metadata("image_type", "sample")
    .with_multimodal_metadata("processing_pipeline", "example");
    
    println!("✅ Created multimodal document:");
    println!("   - Primary modality: {}", multimodal_doc.primary_modality);
    println!("   - Has media content: {}", multimodal_doc.media_content.is_some());
    println!("   - Modalities: {:?}", multimodal_doc.get_modalities());
    println!("   - Total media size: {} bytes", multimodal_doc.total_media_size());
    
    // Example 5: Create multimodal node
    println!("\n🔗 Example 5: Creating multimodal node");
    let chunk_info = cheungfun_core::types::ChunkInfo::new(0, 18, 0); // "Image node content" is 18 chars
    let base_node = Node::new("Image node content", uuid::Uuid::new_v4(), chunk_info);
    let multimodal_node = MultimodalNode::with_media(
        base_node,
        ModalityType::Image,
        image_content.clone(),
    )
    .with_modality_feature("image_category", "sample")
    .with_modality_confidence(ModalityType::Image, 0.95);
    
    println!("✅ Created multimodal node:");
    println!("   - Node ID: {}", multimodal_node.base.id);
    println!("   - Modality: {}", multimodal_node.modality);
    println!("   - Has media: {}", multimodal_node.media_content.is_some());
    println!("   - Primary confidence: {:.2}", multimodal_node.get_primary_confidence().unwrap_or(0.0));
    
    // Example 6: Demonstrate format detection
    println!("\n🔍 Example 6: Format detection");
    let unknown_data = create_sample_image_data();
    let detected_format = cheungfun_multimodal::utils::detect_format(&unknown_data, Some("test.png")).await;
    println!("✅ Detected format: {}", detected_format);
    
    let modality = cheungfun_multimodal::utils::detect_modality_from_path("image.jpg");
    println!("✅ Detected modality from path: {:?}", modality);
    
    // Example 7: Utility functions
    println!("\n🛠️  Example 7: Utility functions");
    let complexity = cheungfun_multimodal::utils::estimate_processing_complexity(&image_content);
    println!("✅ Processing complexity estimate: {:.2}", complexity);
    
    let requires_preprocessing = cheungfun_multimodal::utils::requires_preprocessing(&image_content);
    println!("✅ Requires preprocessing: {}", requires_preprocessing);
    
    let batch_size = cheungfun_multimodal::utils::recommended_batch_size(ModalityType::Image);
    println!("✅ Recommended batch size: {}", batch_size);
    
    // Example 8: Content validation
    println!("\n✅ Example 8: Content validation");
    match cheungfun_multimodal::utils::validate_media_content(&image_content) {
        Ok(()) => println!("✅ Content validation passed"),
        Err(e) => println!("❌ Content validation failed: {}", e),
    }
    
    println!("\n🎉 Image processing example completed successfully!");
    println!("   Check out other examples for audio, video, and cross-modal operations.");
    
    Ok(())
}

/// Create sample image data (a simple PNG).
fn create_sample_image_data() -> Vec<u8> {
    // This is a minimal 1x1 pixel PNG image (red pixel)
    vec![
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk header
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // 1x1 dimensions
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xDE, // IHDR data + CRC
        0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, // IDAT chunk header
        0x08, 0x99, 0x01, 0x01, 0x00, 0x00, 0xFF, 0xFF, // IDAT data (compressed)
        0x00, 0x00, 0x00, 0x02, 0x00, 0x01, 0xE2, 0x21, 0xBC, 0x33, // IDAT data + CRC
        0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, // IEND chunk header
        0xAE, 0x42, 0x60, 0x82, // IEND CRC
    ]
}

/// Demonstrate advanced image operations (when image support is enabled).
#[cfg(feature = "image-support")]
async fn demonstrate_advanced_operations() -> Result<()> {
    use cheungfun_multimodal::processors::ImageProcessor;
    use cheungfun_multimodal::traits::FeatureExtractor;
    
    println!("\n🚀 Advanced Image Operations");
    println!("============================");
    
    let processor = ImageProcessor::builder()
        .unlimited_dimensions()
        .default_quality(0.95)
        .preserve_exif(true)
        .build();
    
    let image_data = create_sample_image_data();
    let content = MediaContent::from_bytes(image_data, MediaFormat::Png);
    
    // Extract specific features
    let extractable_features = processor.extractable_features(ModalityType::Image);
    println!("📋 Extractable features: {:?}", extractable_features);
    
    for feature_name in &extractable_features {
        if let Ok(Some(value)) = processor.extract_feature(&content, feature_name).await {
            println!("   - {}: {}", feature_name, value);
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sample_image_data() {
        let data = create_sample_image_data();
        assert!(!data.is_empty());
        assert!(data.starts_with(&[0x89, 0x50, 0x4E, 0x47])); // PNG signature
    }
    
    #[tokio::test]
    async fn test_image_content_creation() {
        let data = create_sample_image_data();
        let content = MediaContent::from_bytes(data, MediaFormat::Png);
        
        assert_eq!(content.format, MediaFormat::Png);
        assert_eq!(content.modality_type(), ModalityType::Image);
        assert!(content.estimated_size().unwrap() > 0);
    }
    
    #[tokio::test]
    async fn test_multimodal_document_creation() {
        let data = create_sample_image_data();
        let content = MediaContent::from_bytes(data, MediaFormat::Png);
        let doc = Document::new("Test document");
        
        let multimodal_doc = MultimodalDocument::with_media(
            doc,
            ModalityType::Image,
            content,
        );
        
        assert_eq!(multimodal_doc.primary_modality, ModalityType::Image);
        assert!(multimodal_doc.media_content.is_some());
        assert!(multimodal_doc.has_modality(ModalityType::Image));
    }
}
