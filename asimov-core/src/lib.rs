//! # Asimov
//!
//! High performance LLM I/O.

mod db;
pub mod error;
mod io;
pub mod models;
pub mod tokenizers;

pub mod prelude {
    pub use crate::db::hora::HoraDb;
    pub use crate::db::namespace::Namespace;
    #[cfg(feature = "qdrant")]
    pub use crate::db::qdrant::Qdrant;
    pub use crate::db::space::VectorSpace;
    pub use crate::error::{AsimovError, Result};
    pub use crate::io::output::RawString;
    pub use crate::io::{Embeddable, Input};
    #[cfg(feature = "openai")]
    pub use crate::models::openai::*;
    pub use crate::models::{Embed, Generate};
    pub use crate::{lines, prompt};
    pub use asimov_derive::asimov;
    pub use tera;
}

pub use prelude::*;
