use asimov::prelude::*;
use futures::StreamExt;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct OvermindCell {
    id: String,
    specialty: String,
    description: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    let gpt4 = OpenAiLlm::builder()
        .model("gpt-4".to_string())
        .temperature(0.0)
        .build();

    let prompt = prompt!(lines! {
            "Please generate a stream of 20 items. Come up with a unique alphanumeric ID for each one.",
            "Each item should have a speciality from the following: portfolio analyst, inventory manager, data scientist, quality assurance",
            "Each item should have a creative, unique and funny description.",
            "it should be in the format: {\"id\": \"IMNu0lW\", \"specialty\": \"data scientist\", \"description\": \"lead data scientist for the trisolaran empire.\"}",
            "{\"id\": \"dKYNbFo\", \"specialty\": \"portfolio analyst\", \"description\": \"atreides atomics inventory management lead\"}"
    });

    let mut stream: StreamedOutput<OvermindCell> = gpt4.generate(prompt).await?;

    while let Some(item) = stream.next().await {
        let item = item.expect("Failed to deserialize item");
        println!("{:?}", item);
    }

    Ok(())
}
