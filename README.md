# Asimov: Build blazingly fast LLM applications in Rust 


<img src="asimov.png" width="500">

## Overview

`asimov` is a Rust library to build high performance LLM based applications. The crate is divided into the following modules:
1. `io` Utilities for providing structured 

## Quickstart

Here's a simple example using `asimov` to generate a response from an LLM. See the [examples](  https://github.com/overmindai/asimov/tree/master/examples) directory for more examples.

```rust

use asimov::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    let gpt35 = OpenAiLlm::default();
    let response: RawString = gpt35.generate("How are you doing").await?;

    println!("Response: {}", response.0);

    Ok(())
}
```

### Run examples
1. Structured parsing from streams
```
cargo run --example structured_streaming --features openai -- --nocapture
```

2. Basic streaming
```
cargo run --example streaming --features openai -- --nocapture
```

3. Chaining two LLMs
```
cargo run --example simple_chain --features openai
```

4. Few shot generation
```
cargo run --example simple_fewshot --features openai 
```


## Install

Add asimov as a dependency in your `Cargo.toml`

```
cargo add asimov
```
or 
```toml
[dependencies]
asimov = "0.1.0"
```

## Build from source

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

1. I/O module
```bash
cargo test io
```

2. Vector DB module
```bash
cargo test db
```

3. Tokenizers module
```bash
cargo test tokenizers
```

4. Models module
```bash
cargo test models features --openai
```
### Features
The following optional features can be enabled:
* `openai` Enables use of the `async-openai` crate.
* `qdrant` Enables the use of the `qdrant-client` crate.

To enable a feature, use the `--features` flag when building or running:

```bash
cargo build --features "openai qdrant"
```
