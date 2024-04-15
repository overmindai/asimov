pub mod capabilities;
#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "openai")]
use self::openai::{OpenAiEmbedding, OpenAiLlm};

pub use capabilities::{Embed, Generate};
