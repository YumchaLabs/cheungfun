fn main() {
    println!("Hello from simple test!");

    #[cfg(feature = "fastembed")]
    println!("FastEmbed feature is enabled");

    #[cfg(feature = "candle")]
    println!("Candle feature is enabled");

    #[cfg(not(any(feature = "fastembed", feature = "candle")))]
    println!("No embedding features enabled");
}
