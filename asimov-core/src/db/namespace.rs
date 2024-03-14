use serde::{Deserialize, Serialize};
use std::{fmt::Display, ops::Deref};

use crate::{AsimovError, Result};

/// Namespaces are used to group/cagegorize related items in the VectorSpace.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct Namespace(pub String);

impl Namespace {
    /// Creates a new new `Namespace` from string.
    pub fn new(name: String) -> Result<Self> {
        // Validate for non-empty string
        if name.is_empty() {
            return Err(AsimovError::Namespace("Namespace cannot be empty".into()));
        }

        if !name.is_ascii() {
            return Err(AsimovError::Namespace(
                "Only ascii characters are allowed".into(),
            ));
        }
        // Additional validations can be added here
        // Example: checking for maximum length, etc.
        Ok(Namespace(name))
    }
}

impl TryFrom<String> for Namespace {
    type Error = AsimovError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Namespace::new(value)
    }
}

impl TryFrom<&str> for Namespace {
    type Error = AsimovError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Namespace::new(value.to_string())
    }
}

impl Display for Namespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Namespace> for String {
    fn from(value: Namespace) -> Self {
        value.0
    }
}

impl From<&Namespace> for String {
    fn from(value: &Namespace) -> Self {
        value.0.clone()
    }
}

impl Deref for Namespace {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}
