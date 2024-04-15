/// LLM chaining example:
/// 1. Use the first LLM to determine the sentiment of a given input.
/// 2. Use the second LLM to summarize the sentiment.
use asimov::prelude::*;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
struct Sentiment {
    sentiment: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    // First LLM for sentiment analysis
    let sentiment_analyzer = OpenAiLlm::builder()
        .temperature(0.0)
        .model("gpt-3.5-turbo".to_string())
        .build();

    // Second LLM for summarizing sentiment
    let summarizer = OpenAiLlm::builder()
        .model("gpt-3.5-turbo".to_string())
        .temperature(0.0)
        .build();

    let input = "The product is of high quality and the customer service is excellent.";

    let sentiment_prompt = prompt!(
        lines! {
            "Return the sentiment, true represents positive, false is negative.",
            "The output format should be like so: {\"sentiment\": true}",
            "Q: {{input}}",
            "A:"
        },
        input
    );

    let sentiment: Sentiment = sentiment_analyzer.generate(&sentiment_prompt).await?;
    let sent: bool = sentiment.sentiment;

    println!("Sentiment: {}", sent);

    let summary_prompt = prompt!(
        lines! {
            "The sentiment of the text '{{input}}' is {{sent}}.",
            "Summarize it"
        },
        input,
        sent
    );

    // Use the second LLM to summarize the sentiment
    let summary: RawString = summarizer.generate(&summary_prompt).await?;
    println!("Summary: {}", summary.0);

    Ok(())
}
