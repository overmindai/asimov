use crate::{error::AsimovError, Input, Result};

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Vec<usize>;
    fn decode(&self, tokens: &[usize]) -> Result<String, AsimovError>;
    fn length(&self, text: &str) -> usize;

    fn num_tokens<I: Input + ?Sized>(&self, query: &I) -> Result<usize> {
        let s = query.render()?;
        Ok(self.encode(s.as_str()).len())
    }

    fn tokenize<I: Input + ?Sized>(&self, query: &I) -> Result<Vec<usize>> {
        let s = query.render()?;
        Ok(self.encode(s.as_str()))
    }
}

pub mod openai;
