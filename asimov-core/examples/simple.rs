/// Simple example of using asimov to generate code.
use asimov::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    let gpt35 = OpenAiLlm::default();
    let response: RawString = gpt35.generate("How are you doing").await?;

    println!("Response: {}", response.0);

    Ok(())
}
