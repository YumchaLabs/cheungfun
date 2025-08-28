use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    test: bool,
}

fn main() {
    let _args = Args::parse();
    println!("Test compilation successful!");
}
