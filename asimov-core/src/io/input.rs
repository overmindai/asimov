use crate::error::{AsimovError, Result};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::hash::{Hash, Hasher};

use twox_hash::XxHash64;

/// `Input` represents any object that can be provided to a LLM as input.
pub trait Input: Send + Sync {
    /// returns a string representation of the value suitable for consumption by a LLM.
    fn render(&self) -> Result<String>;
    /// generates a unique hash of the input.
    fn hash(&self) -> Result<u64> {
        let s = &self.render()?;

        let mut h = XxHash64::default();
        Hash::hash(&s, &mut h);

        Ok(h.finish())
    }
}

/// `Embeddable` represents an object that can be embedded. Used in `capabilities::Embed``
/// If used with the `asimov` attribute macro, it will automatically implement `Embeddable` for the struct.
/// If no key is provided, the struct itself will be used as the key.
pub trait Embeddable: Input + Sync + Send + 'static {
    type Key: Input + Serialize + Clone + for<'de> Deserialize<'de> + Send + Sync;

    fn key(&self) -> Self::Key;
}

impl Input for Value {
    fn render(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| AsimovError::Input(e.to_string()))
    }
}

impl<T: Input> Input for &T {
    fn render(&self) -> Result<String> {
        (*self).render()
    }
}

impl<T: Input> Input for Vec<T> {
    fn render(&self) -> Result<String> {
        let mut s = String::new();
        for (index, item) in self.iter().enumerate() {
            if index > 0 {
                s.push_str("===\n");
            }
            s.push_str(&item.render()?);
        }
        Ok(s)
    }
}

impl Input for String {
    fn render(&self) -> Result<String> {
        Ok(self.clone())
    }
}

impl Input for &str {
    fn render(&self) -> Result<String> {
        Ok(self.to_string())
    }
}

impl Input for bool {
    fn render(&self) -> Result<String> {
        Ok(self.to_string())
    }
}

impl Input for i32 {
    fn render(&self) -> Result<String> {
        Ok(self.to_string())
    }
}

impl Input for i64 {
    fn render(&self) -> Result<String> {
        Ok(self.to_string())
    }
}

impl Input for f32 {
    fn render(&self) -> Result<String> {
        Ok(self.to_string())
    }
}

impl Input for f64 {
    fn render(&self) -> Result<String> {
        Ok(self.to_string())
    }
}

impl<T: Input> Input for Option<T> {
    fn render(&self) -> Result<String> {
        match self {
            Some(v) => v.render(),
            None => Ok("".to_string()),
        }
    }
}

impl Input for Box<&str> {
    fn render(&self) -> Result<String> {
        Ok((**self).to_string())
    }
}
#[macro_export]
macro_rules! lines {
    // Base case for the recursive macro
    ($($str:expr),*) => {
        {
            use std::fmt::Write;
            let mut result = String::new();
            $(writeln!(result, "{}", $str).expect("Error writing to string");)*
            result
        }
    };
}

#[macro_export]
macro_rules! prompt {

    ($template:expr) => {
        {
            let tera_context = tera::Context::new(); // Create an empty context
            let tera = tera::Tera::one_off(&$template, &tera_context, true)
                .expect("Failed to render template");
            tera
        }
    };
    // Existing pattern for template string with identifiers
    ($template:expr, $($idents:expr),*) => {
        {
            let mut tera_context = tera::Context::new();
            $(
                tera_context.insert(stringify!($idents), &$idents.render().unwrap());
            )*
            let tera = tera::Tera::one_off(&$template, &tera_context, true)
                .expect("Failed to render template");
            tera
        }
    };
}

impl Embeddable for String {
    type Key = String;

    fn key(&self) -> Self::Key {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizers::Tokenizer;
    use crate::Embeddable;
    use serde::Deserialize;

    #[derive(Serialize, Deserialize, Clone)]

    struct CustomType {
        key: String,
        name: String,
    }

    impl Input for CustomType {
        fn render(&self) -> Result<String> {
            Ok(self.name.clone())
        }
    }

    impl Embeddable for CustomType {
        type Key = String;

        fn key(&self) -> Self::Key {
            self.key.clone()
        }
    }

    #[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
    struct CustomType2 {
        name: String,
        phone: String,
        address: String,
    }

    impl Input for CustomType2 {
        fn render(&self) -> Result<String> {
            Ok(self.name.clone())
        }
    }

    impl Embeddable for CustomType2 {
        type Key = String;

        fn key(&self) -> Self::Key {
            self.name.clone()
        }
    }

    #[derive(Serialize, Deserialize, Clone)]

    struct CustomType3 {
        name: String,
        internal: CustomType2,
    }

    impl Input for CustomType3 {
        fn render(&self) -> Result<String> {
            Ok(self.name.clone())
        }
    }

    impl Embeddable for CustomType3 {
        type Key = String;

        fn key(&self) -> Self::Key {
            self.name.clone()
        }
    }

    #[test]
    fn test_embed() {
        let custom_type = CustomType2 {
            name: "custom object".to_string(),
            phone: "1234567890".to_string(),
            address: "1234 Main St".to_string(),
        };

        assert_eq!(custom_type.key().render().unwrap(), "custom object");
    }

    #[test]
    fn test_input() {
        let custom_type = CustomType {
            key: "key".to_string(),
            name: "custom object".to_string(),
        };

        let another_custom_type = CustomType {
            key: "key".to_string(),
            name: "another custom object".to_string(),
        };

        let context = prompt!(
            lines! {
                "do this task using: {{custom_type}}",
                "Another task using: {{another_custom_type}}"
            },
            custom_type,
            another_custom_type
        );

        assert_eq!(
            context,
            "do this task using: key, custom object\nAnother task using: key, another custom object\n"
        );
    }

    #[test]
    fn test_with_tokenizer() {
        let custom_type = CustomType {
            key: "key".to_string(),
            name: "my custom object".to_string(),
        };

        let context = prompt!(
            lines! {"do this task", "Another task", "{{custom_type}}"},
            custom_type
        );

        let tokenizer = crate::tokenizers::openai::openai_tiktoken::OpenAiTiktoken::new();
        assert_eq!(tokenizer.num_tokens(&context).unwrap(), 13);
    }

    #[test]
    fn macro_prompt() {
        let ident1 = "23.4,56.4,56.3,23.4,443.2,456.4,456.2,42.5";
        let ident2 = "1";
        let ident3 = "1";

        let rendered = prompt!(
            lines! {
                "You are a data scientist. You have been given the following values in csv: ",
                "data: {{ ident1 }}",
                "Calculate the median of the above values.",
                "number 1: {{ ident2 }}",
                "number 2: {{ ident3 }}",
                "Calculate the sum of above 2 numbers."
            },
            ident1,
            ident2,
            ident3
        );

        println!("{}", rendered);
    }

    #[test]
    fn test_nested_embed() {
        let custom_type = CustomType3 {
            name: "custom object".to_string(),
            internal: CustomType2 {
                name: "custom object".to_string(),
                phone: "1234567890".to_string(),
                address: "1234 Main St".to_string(),
            },
        };

        println!("{}", custom_type.key().render().unwrap());
    }
}
