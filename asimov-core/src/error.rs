use thiserror::Error;

#[derive(Error, Debug)]
pub enum AsimovError {
    #[error("Context error: {0}")]
    Input(String),
    #[error("Output error: {0}")]
    Output(String),
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    #[error("Namespace error: {0}")]
    Namespace(String),
    #[cfg(feature = "openai")]
    #[error("OpenAI error")]
    OpenAI(#[from] async_openai::error::OpenAIError),
    #[error("Parsing Error")]
    ParsingError(#[from] serde_json::error::Error),
    #[error("Model error: {0}")]
    Model(String),
    #[error("Key {0} is already present")]
    KeyCollision(String),
    #[error("Key {0} not found")]
    KeyNotFound(String),
    #[error("Invalid namespace")]
    InvalidNamespace,
    #[error("{0}")]
    Hora(String),
    #[error("{0}")]
    Qdrant(String),
    #[error("{0}")]
    Anyhow(#[from] anyhow::Error),
    #[error("VectorDb error: {0}")]
    VectorDb(String),
    #[error("Few shot error: {0}")]
    FewShot(String),
    #[error("unknown  error {0}")]
    Unknown(String),
}

pub type Result<T, E = AsimovError> = std::result::Result<T, E>;
