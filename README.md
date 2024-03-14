# asimov

<img src="asimov.png" width="300">

High performance I/O for LLMs (large language models).

Asimov is a crate for creating and managing input and output for large language models.

## Getting Started

To get started with asimov:

Add asimov as a dependency in your `Cargo.toml`:

```toml
[dependencies]
asimov = "0.1.0"
```

### Example

Here's a simple example of using asimov to generate a summary of a PDF.

```rust
use asimov::{
    models::{Generate, ModelName},
    Input, RawString,
};

#[cfg(feature = "openai")]
use asimov::models::openai::OpenAiLlm;

#[cfg(feature = "pdf")]
async fn pdf_text(uri: &str) -> String {
    let pdf_stream = reqwest::get(uri).await.unwrap().text().await.unwrap();
    let bytes = std::fs::read(pdf_stream).unwrap();
    let out = pdf_extract::extract_text_from_mem(&bytes).unwrap();
    out
}

#[tokio::main]
async fn main() {
    dotenvy::dotenv().unwrap();
    // Second LLM for summarizing sentiment
    let summarizing_llm = OpenAiLlm::builder()
        .model(ModelName::Gpt35Turbo)
        .temperature(0.0)
        .build();

    let summary_input = format!(
        "Summarize the text below from a pdf. Keep it brief in bullet points. \n{}",
        pdf_text("https://riak.com/assets/bitcask-intro.pdf").await,
    );

    let summary: RawString = summarizing_llm.generate(summary_input).await.unwrap();

    println!("Summary\n {summary}");
}

```

# Building from source

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) 1.54.0 or later
- [Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html) (comes with Rust)


### Clone the repository:
```bash
git clone https://github.com/yourusername/asimov.git
cd asimov
```

### Build the project

```bash
cargo build
```

### Run Tests
```bash
cargo test
```

### Features
The project has optional features that can be enabled:
* `openai`: Enables use of the `async-openai` crate.
* `qdrant`: Enables the use of the `qdrant-client` crate.
* `pdf`: Enables APIs for prompt generation via PDFs as a source.
* `csv`: Enables APIs for prompt generation via csv as a source.

To enable a feature, use the `--features` flag when building or running:

```bash
cargo build --features "openai qdrant"
```
