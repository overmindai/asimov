pub mod capabilities;
#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "openai")]
use self::openai::{OpenAiEmbedding, OpenAiLlm};
use crate::tokenizers;
pub use capabilities::{Embed, Generate};
use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};
