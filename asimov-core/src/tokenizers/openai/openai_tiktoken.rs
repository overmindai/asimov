use tiktoken_rs::{p50k_base, CoreBPE};

use crate::tokenizers::Tokenizer;

pub struct OpenAiTiktoken {
    bpe: CoreBPE,
}

impl Default for OpenAiTiktoken {
    fn default() -> Self {
        Self {
            bpe: p50k_base().unwrap(),
        }
    }
}

impl OpenAiTiktoken {
    pub fn new() -> Self {
        Default::default()
    }
}

impl Tokenizer for OpenAiTiktoken {
    fn encode(&self, text: &str) -> Vec<usize> {
        self.bpe.encode_with_special_tokens(text)
    }

    fn decode(&self, tokens: &[usize]) -> Result<String, crate::error::AsimovError> {
        let result = self.bpe._decode_native(tokens);
        String::from_utf8(result).map_err(|_| {
            crate::error::AsimovError::Tokenizer("Returned sequence is not utf-8 encoded".into())
        })
    }

    fn length(&self, text: &str) -> usize {
        self.encode(text).len()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_gpt_tokenizer() {
        let examples: Vec<(String, usize)> = vec![
            ("This is a test".to_string(), 4),
            ("Another test".to_string(), 2),
        ];
        let gpt = OpenAiTiktoken::new();

        for (text, length) in examples.into_iter() {
            assert_eq!(gpt.length(&text), length)
        }
    }
}
