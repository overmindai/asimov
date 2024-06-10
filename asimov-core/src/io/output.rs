use crate::error::Result;
use derive_more::{Deref, DerefMut, Display, From};
use std::pin::Pin;
use std::task::Poll;

use futures::Stream;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

type ItemStream<T> = Pin<Box<dyn Stream<Item = T> + Send>>; // Ensure the Stream is Send

/// Building block for streaming LLM responses.
///
/// Holds a stream of tokens generated by the LLM.
pub struct TokenStream {
    stream: ItemStream<String>,
}

impl TokenStream {
    pub fn new(stream: impl Stream<Item = String> + Send + 'static) -> Self {
        Self {
            stream: Box::pin(stream),
        }
    }
}

impl Stream for TokenStream {
    type Item = String;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.get_mut().stream).poll_next(cx)
    }
}

/// This struct lets us stream parsed elements
pub struct JsonStream<D: DeserializeOwned + Send> {
    stream: json_stream::JsonStream<D, ItemStream<Result<Vec<u8>>>>,
}

impl<D: DeserializeOwned + Send> JsonStream<D> {
    pub fn new(stream: ItemStream<Result<Vec<u8>>>) -> Self {
        Self {
            stream: json_stream::JsonStream::new(stream),
        }
    }
}

impl<D: DeserializeOwned + Send> Stream for JsonStream<D> {
    type Item = Result<D>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.get_mut().stream).poll_next(cx)
    }
}

/// String new type that does *not* implement the Deserialize trait.
///
/// If we use String directly, the `Generate<String>` will
/// use `Generate<T: Deserialize>` and fail if the response is not
/// quotation-delimited.
///
/// We use the `derive_more` trait to auto implement traits such as
/// `Display`, `From`, `Deref`, `DerefMut`, making our lives easier.
#[derive(Display, From, Clone, Deref, DerefMut, Debug, PartialEq, Eq, Serialize)]
pub struct RawString(pub String);

impl RawString {
    pub fn new(s: String) -> Self {
        Self(s)
    }
}

/// `StreamedOutput` provides a general abstraction over streamed responses.
///
/// To generate a streamed response, you only need to wrap the result type
/// into `Streamed`.
///
/// Since parsing is fallible, streamed results are wrapped into a `Result` enum.
///
/// You can parse JSONL content out-of-the-box for types
/// that implement `Deserialize`.
///
/// Note that `Streamed<String>` will expect newline-separated quoted strings.
/// Moreover, `Streamed<RawString>` has a special behavior, and simply
/// returns the token stream wrapped in an `Ok`.
///
/// Should you want to access the token stream directly, you should use
/// [`TokenStream`] as a result type. [`TokenStream`] is a stream of `String`
/// objects.
pub struct StreamedOutput<T> {
    stream: ItemStream<Result<T>>,
}

impl<T> StreamedOutput<T> {
    pub fn new(stream: impl Stream<Item = Result<T>> + Send + 'static) -> Self {
        Self {
            stream: Box::pin(stream),
        }
    }
}

impl<T> Stream for StreamedOutput<T> {
    type Item = Result<T>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.get_mut().stream).poll_next(cx)
    }
}
