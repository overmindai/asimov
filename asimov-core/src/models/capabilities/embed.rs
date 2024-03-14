use crate::{error::Result, tokenizers::Tokenizer, Input};
use async_trait::async_trait;

/// Trait that defines the behavior of embedding models.
#[async_trait]
pub trait Embed: Send + Sync {
    type Tokenizer: Tokenizer;
    const DIM: u32;

    /// Embed an object that implements the [`Embeddable`] trait.
    async fn embed<I: Input + ?Sized>(&self, input: &I) -> Result<Vec<f32>>;
}
