/// Structured output parsing from a stream
use asimov::prelude::*;
use futures::StreamExt;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
struct Item {
    value: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    let gpt35 = OpenAiLlm::builder().temperature(0.0).build();

    let prompt = r#"Output the following items in a stream twice, each item separated by a new line: {"value": "foo"}"#;

    let mut stream: StreamedOutput<Item> = gpt35.generate(prompt).await?;

    while let Some(item) = stream.next().await {
        let item = item.expect("Failed to deserialize item");
        println!("{:?}", item);
    }

    Ok(())
}
