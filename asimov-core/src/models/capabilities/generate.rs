use crate::error::Result;
use crate::io::Input;
// use crate::io::StreamedOutput;

use async_trait::async_trait;

/// Generate a result from the LLM.
///
/// To stream the response, you should wrap the expected object inside
/// [`StreamedOutput`](crate::io::StreamedOutput).
///
/// Otherwise, the result will be parsed "synchronously":
/// the full LLM output is awaited and parsed as a whole.

#[async_trait]
pub trait Generate<T = String> {
    async fn generate(&self, input: impl Input) -> Result<T>;
}
