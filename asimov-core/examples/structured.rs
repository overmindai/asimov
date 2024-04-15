/// Structured output parsing example
use asimov::prelude::*;
use futures::StreamExt;
use serde::Deserialize;
#[derive(Deserialize, Debug)]
struct Person {
    name: String,
    age: u8,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    let gpt35 = OpenAiLlm::builder().temperature(0.0).build();

    let prompt = r#"Output the following verbatim {"name": "Alice", "age": 30}"#;

    let person: Person = gpt35.generate(prompt).await?;

    println!("{:?}", person);

    Ok(())
}
