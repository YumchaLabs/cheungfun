//! Audio processing example.
//!
//! This example demonstrates the audio processing capabilities of the
//! cheungfun-multimodal crate, including:
//! - Loading audio files from various sources
//! - Format detection and validation
//! - Audio feature extraction
//! - Format conversion
//! - Audio processing operations

use cheungfun_multimodal::processors::audio::{AudioLoader, AudioConverter, AudioFeatureExtractor, AudioProcessor};
use cheungfun_multimodal::traits::{MediaProcessor, ProcessingOptions};
use cheungfun_multimodal::traits::media_processor::AudioProcessingOptions;
use cheungfun_multimodal::types::{MediaFormat, MediaContent, MediaData, MediaMetadata};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("🎵 Cheungfun Audio Processing Example");
    println!("=====================================\n");
    
    // Example 1: Create and configure audio loader
    println!("📁 Example 1: Audio Loader Configuration");
    let loader = AudioLoader::builder()
        .max_file_size(Some(50 * 1024 * 1024)) // 50MB limit
        .validate_format(true)
        .supported_formats(vec![
            MediaFormat::Mp3,
            MediaFormat::Wav,
            MediaFormat::Flac,
            MediaFormat::Ogg,
        ])
        .build();
    
    println!("✅ Audio loader configured:");
    println!("   - Max file size: 50MB");
    println!("   - Format validation: enabled");
    println!("   - Supported formats: MP3, WAV, FLAC, OGG");
    
    // Example 2: Create dummy audio content for demonstration
    println!("\n🎧 Example 2: Creating Audio Content");
    let audio_content = create_dummy_wav_content();
    
    println!("✅ Created dummy WAV audio content:");
    println!("   - Format: {}", audio_content.format);
    println!("   - Size: {} bytes", audio_content.estimated_size().unwrap_or(0));
    
    // Example 3: Audio feature extraction
    println!("\n🔍 Example 3: Audio Feature Extraction");
    let feature_extractor = AudioFeatureExtractor::builder()
        .extract_spectral_features(true)
        .extract_tempo_features(true)
        .extract_harmonic_features(true)
        .build();
    
    match feature_extractor.extract_features(&audio_content).await {
        Ok(features) => {
            println!("✅ Extracted audio features:");
            for (key, value) in features.iter() {
                println!("   - {}: {}", key, value);
            }
        }
        Err(e) => {
            println!("❌ Failed to extract features: {}", e);
        }
    }
    
    // Example 4: Audio format conversion
    println!("\n🔄 Example 4: Audio Format Conversion");
    let converter = AudioConverter::builder()
        .default_sample_rate(44100)
        .default_channels(2)
        .normalize_audio(true)
        .output_quality(0.9)
        .build();
    
    match converter.convert_format(&audio_content, MediaFormat::Mp3).await {
        Ok(converted_content) => {
            println!("✅ Converted audio format:");
            println!("   - Original format: {}", audio_content.format);
            println!("   - New format: {}", converted_content.format);
            println!("   - Original size: {} bytes", audio_content.estimated_size().unwrap_or(0));
            println!("   - New size: {} bytes", converted_content.estimated_size().unwrap_or(0));
        }
        Err(e) => {
            println!("❌ Failed to convert format: {}", e);
        }
    }
    
    // Example 5: Audio processing with options
    println!("\n⚙️  Example 5: Audio Processing with Options");
    let processor = AudioProcessor::builder()
        .max_duration(300.0) // 5 minutes
        .default_sample_rate(48000)
        .normalize_audio(true)
        .build();
    
    let processing_options = ProcessingOptions {
        target_format: Some(MediaFormat::Flac),
        quality: Some(0.95),
        audio_options: Some(AudioProcessingOptions {
            target_sample_rate: Some(48000),
            target_channels: Some(1), // Convert to mono
            normalize: true,
            trim_start: Some(0.0),
            trim_duration: Some(30.0), // Trim to 30 seconds
            target_bitrate: None,
            extract_speech: false,
        }),
        ..Default::default()
    };
    
    match processor.process(&audio_content, &processing_options).await {
        Ok(processed_content) => {
            println!("✅ Processed audio:");
            println!("   - Original format: {}", audio_content.format);
            println!("   - Processed format: {}", processed_content.format);
            println!("   - Processing applied: sample rate conversion, mono conversion, normalization");
        }
        Err(e) => {
            println!("❌ Failed to process audio: {}", e);
        }
    }
    
    // Example 6: Audio sample rate conversion
    println!("\n📊 Example 6: Sample Rate Conversion");
    match converter.convert_sample_rate(&audio_content, 22050).await {
        Ok(resampled_content) => {
            println!("✅ Converted sample rate:");
            println!("   - Target sample rate: 22050 Hz");
            println!("   - Conversion completed successfully");
        }
        Err(e) => {
            println!("❌ Failed to convert sample rate: {}", e);
        }
    }
    
    // Example 7: Channel conversion (stereo to mono)
    println!("\n🔊 Example 7: Channel Conversion");
    match converter.convert_channels(&audio_content, 1).await {
        Ok(mono_content) => {
            println!("✅ Converted to mono:");
            println!("   - Original channels: 2 (stereo)");
            println!("   - New channels: 1 (mono)");
        }
        Err(e) => {
            println!("❌ Failed to convert channels: {}", e);
        }
    }
    
    // Example 8: Audio normalization
    println!("\n📈 Example 8: Audio Normalization");
    match converter.normalize_audio(&audio_content).await {
        Ok(normalized_content) => {
            println!("✅ Audio normalized:");
            println!("   - Normalization applied to balance audio levels");
        }
        Err(e) => {
            println!("❌ Failed to normalize audio: {}", e);
        }
    }
    
    // Example 9: Audio trimming
    println!("\n✂️  Example 9: Audio Trimming");
    match converter.trim_audio(&audio_content, 5.0, Some(10.0)).await {
        Ok(trimmed_content) => {
            println!("✅ Audio trimmed:");
            println!("   - Start time: 5.0 seconds");
            println!("   - Duration: 10.0 seconds");
            println!("   - Trimmed audio created successfully");
        }
        Err(e) => {
            println!("❌ Failed to trim audio: {}", e);
        }
    }
    
    // Example 10: Load from bytes
    println!("\n💾 Example 10: Loading Audio from Bytes");
    let audio_bytes = create_dummy_mp3_bytes();
    match loader.load_from_bytes(audio_bytes, Some("test.mp3")).await {
        Ok(loaded_content) => {
            println!("✅ Loaded audio from bytes:");
            println!("   - Format: {}", loaded_content.format);
            println!("   - Size: {} bytes", loaded_content.estimated_size().unwrap_or(0));
        }
        Err(e) => {
            println!("❌ Failed to load from bytes: {}", e);
        }
    }
    
    println!("\n🎉 Audio processing examples completed!");
    println!("\nNote: This example uses dummy audio data for demonstration.");
    println!("In a real application, you would load actual audio files.");
    
    Ok(())
}

/// Create dummy WAV audio content for demonstration.
fn create_dummy_wav_content() -> MediaContent {
    // Create a minimal WAV file header
    let mut wav_data = Vec::new();
    
    // RIFF header
    wav_data.extend_from_slice(b"RIFF");
    wav_data.extend_from_slice(&[36, 0, 0, 0]); // File size - 8
    wav_data.extend_from_slice(b"WAVE");
    
    // Format chunk
    wav_data.extend_from_slice(b"fmt ");
    wav_data.extend_from_slice(&[16, 0, 0, 0]); // Subchunk1Size
    wav_data.extend_from_slice(&[1, 0]); // AudioFormat (PCM)
    wav_data.extend_from_slice(&[2, 0]); // NumChannels (stereo)
    wav_data.extend_from_slice(&[0x44, 0xAC, 0, 0]); // SampleRate (44100)
    wav_data.extend_from_slice(&[0x10, 0xB1, 2, 0]); // ByteRate
    wav_data.extend_from_slice(&[4, 0]); // BlockAlign
    wav_data.extend_from_slice(&[16, 0]); // BitsPerSample
    
    // Data chunk
    wav_data.extend_from_slice(b"data");
    wav_data.extend_from_slice(&[0, 0, 0, 0]); // Subchunk2Size
    
    let mut metadata = MediaMetadata::new();
    metadata.filename = Some("dummy.wav".to_string());
    
    MediaContent {
        data: MediaData::Embedded(wav_data),
        format: MediaFormat::Wav,
        metadata,
        extracted_text: None,
        features: HashMap::new(),
        checksum: None,
    }
}

/// Create dummy MP3 bytes for demonstration.
fn create_dummy_mp3_bytes() -> Vec<u8> {
    // Create a minimal MP3 frame header
    let mut mp3_data = Vec::new();
    
    // MP3 frame sync + header
    mp3_data.extend_from_slice(&[0xFF, 0xFB, 0x90, 0x00]); // MP3 frame header
    
    // Add some dummy data
    mp3_data.extend(vec![0; 1024]);
    
    mp3_data
}
