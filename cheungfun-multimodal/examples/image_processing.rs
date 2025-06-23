//! Image processing example.
//!
//! This example demonstrates how to use the multimodal framework to process
//! images, extract features, and perform basic image operations.

use cheungfun_multimodal::prelude::*;

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

/// Create sample image data using PNG crate to generate a valid PNG.
fn create_sample_image_data() -> Vec<u8> {
    use std::io::Cursor;

    // Create a 4x4 pixel RGB image with a simple pattern
    let width = 4u32;
    let height = 4u32;

    // Create RGB data: red, green, blue, white pattern
    let mut rgb_data = Vec::new();
    for y in 0..height {
        for x in 0..width {
            match (x + y) % 4 {
                0 => rgb_data.extend_from_slice(&[255, 0, 0]),   // Red
                1 => rgb_data.extend_from_slice(&[0, 255, 0]),   // Green
                2 => rgb_data.extend_from_slice(&[0, 0, 255]),   // Blue
                _ => rgb_data.extend_from_slice(&[255, 255, 255]), // White
            }
        }
    }

    // Encode as PNG
    let mut png_data = Vec::new();
    {
        let mut cursor = Cursor::new(&mut png_data);
        let mut encoder = png::Encoder::new(&mut cursor, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);

        let mut writer = encoder.write_header().expect("Failed to write PNG header");
        writer.write_image_data(&rgb_data).expect("Failed to write PNG data");
    }

    png_data
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
