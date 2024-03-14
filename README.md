# asimov

<img src="asimov.png" width="700">

High performance I/O for LLMs (large language models).

Asimov is a crate for creating and managing input and output for large language models.

## Getting Started

To get started with asimov:

Add asimov as a dependency in your `Cargo.toml`:

```toml
[dependencies]
asimov = "0.1.0"
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

To enable a feature, use the `--features` flag when building or running:

```bash
cargo build --features "openai qdrant"
```
