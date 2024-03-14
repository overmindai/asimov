use std::marker::PhantomData;

use async_trait::async_trait;
use qdrant_client::prelude::*;
use qdrant_client::prelude::*;
use qdrant_client::qdrant::points_selector::PointsSelectorOneOf;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{
    CreateCollection, PointsIdsList, PointsSelector, SearchPoints, VectorParams, VectorsConfig,
};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::json;

use crate::io::Embeddable;
use crate::{
    error::{AsimovError, Result},
    io::Input,
    models::Embed,
};

use super::space::VectorSpace;

pub struct Qdrant<E: Embed, I: Embeddable> {
    client: QdrantClient,
    llm: E,
    marker: PhantomData<I>,
}

/// The input should be represented as an enum to handle multiple formats.
impl<E, I> Qdrant<E, I>
where
    E: Embed,
    I: Embeddable,
{
    pub fn new(client: QdrantClient, llm: E) -> Self {
        Self {
            client,
            llm,
            marker: PhantomData,
        }
    }
}

#[async_trait]
impl<I, E> VectorSpace for Qdrant<E, I>
where
    E: Embed,
    I: Embeddable + Send + Sync + Serialize + DeserializeOwned + Clone + 'static,
{
    type Item = I;
    async fn namespace_exists(&mut self, namespace: &str) -> Result<bool> {
        let exists = self.client.has_collection(namespace).await?;
        Ok(exists)
    }
    async fn create_namespace(&mut self, namespace: &str) -> Result<()> {
        self.client
            .create_collection(&CreateCollection {
                collection_name: namespace.to_string(),
                vectors_config: Some(VectorsConfig {
                    config: Some(Config::Params(VectorParams {
                        size: E::DIM as u64,
                        distance: Distance::Cosine as i32,
                        ..Default::default()
                    })),
                }),
                ..Default::default()
            })
            .await?;
        Ok(())
    }

    async fn delete_namespace(&mut self, namespace: &str) -> Result<()> {
        self.client.delete_collection(namespace).await?;
        Ok(())
    }

    async fn add_items<It>(&mut self, namespace: &str, items: It) -> Result<()>
    where
        It: IntoIterator<Item = Self::Item> + Send,
        <It as IntoIterator>::IntoIter: Send,
    {
        let mut points = Vec::new();

        for item in items {
            let text = item.render()?;
            let embedding = self.llm.embed(&text).await?;
            let id: u64 = text.hash()?;

            let payload = json!({
                "data": item,
            });

            let point = PointStruct::new(id, embedding, payload.try_into().unwrap());
            points.push(point);
        }
        self.client
            .upsert_points(&namespace.to_string(), None, points, None)
            .await?;

        Ok(())
    }

    async fn delete_item(&mut self, namespace: &str, key: Self::Item) -> Result<()> {
        let text = key.render()?;
        let id: u64 = text.hash()?;

        let points = PointsSelector {
            points_selector_one_of: Some(PointsSelectorOneOf::Points(PointsIdsList {
                ids: vec![id.into()],
            })),
        };

        self.client
            .delete_points(namespace, None, &points, None)
            .await?;

        Ok(())
    }

    async fn knn<K: Input>(&self, namespace: &str, query: &K, k: usize) -> Result<Vec<Self::Item>> {
        let query_clone = query.clone();
        let query = self.llm.embed(&query_clone).await?;
        let response = self
            .client
            .search_points(&SearchPoints {
                collection_name: namespace.to_string(),
                vector: query,
                limit: k as u64,
                with_payload: Some(true.into()),
                ..Default::default()
            })
            .await?;

        let values = response
            .result
            .into_iter()
            .map(|mut s| {
                let item = serde_json::from_value(s.payload.remove("data").unwrap().into())
                    .map_err(|e| {
                        AsimovError::Model(format!("Failed to deserialize payload: {}", e))
                    });
                item
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(values)
    }
}

#[cfg(test)]
mod test {

    use serde::Deserialize;

    use super::*;
    use crate::tokenizers::openai::OpenAiTiktoken;

    struct MockEmbed;

    #[async_trait]
    impl Embed for MockEmbed {
        type Tokenizer = OpenAiTiktoken;
        const DIM: u32 = 128;

        async fn embed<I: Input + ?Sized>(&self, _input: &I) -> Result<Vec<f32>> {
            let embedding = vec![0.0; Self::DIM as usize];
            // Mock embedding logic
            Ok(embedding)
        }
    }

    struct TestQdrant<I: Embeddable>(Qdrant<MockEmbed, I>);

    impl<I: Embeddable + Send + Sync> TestQdrant<I> {
        fn new() -> Self {
            let api_key = std::env::var("QDRANT_API_KEY").expect("QDRANT_API_KEY not set");
            let qdrant_url = std::env::var("QDRANT_URL").expect("QDRANT_URL not set");
            let mut client_builder = QdrantClient::from_url(&qdrant_url).with_api_key(api_key);

            let client = client_builder
                .build()
                .expect("Failed to build Qdrant client");
            let client = Qdrant::new(client, MockEmbed);
            Self(client)
        }
    }

    #[tokio::test]
    async fn test_qdrant() -> Result<()> {
        let mut qdrant = TestQdrant::<String>::new();
        let namespace = "test_namespace";

        let _ = qdrant.0.delete_namespace(namespace).await;
        qdrant.0.create_namespace(namespace).await?;

        let keys = vec![
            "test key 1".to_string(),
            "test key 2".to_string(),
            "test key 3".to_string(),
            "test key 4".to_string(),
            "test key 5".to_string(),
        ];

        qdrant.0.add_items(&namespace, keys).await?;

        let query = "test query".to_string();
        let k = 3;
        let result = qdrant.0.knn(&namespace, &query, k).await?;

        assert_eq!(result.len(), k);

        qdrant.0.delete_namespace(&namespace).await?;

        Ok(())
    }

    #[derive(Clone, Serialize, Deserialize)]
    struct Example {
        text: String,
    }

    impl Example {
        fn new(s: impl Into<String>) -> Self {
            Self { text: s.into() }
        }
    }

    impl Input for Example {
        fn render(&self) -> Result<String> {
            Ok(self.text.clone())
        }
    }

    #[derive(Clone, Serialize, Deserialize)]
    enum Inputs {
        Str(String),
        Ex(Example),
    }

    impl Input for Inputs {
        fn render(&self) -> Result<String> {
            match self {
                Inputs::Ex(e) => e.render(),
                Inputs::Str(s) => s.render(),
            }
        }
    }

    impl Embeddable for Inputs {
        type Key = String;

        fn key(&self) -> &Self::Key {
            match self {
                Inputs::Ex(e) => &e.text,
                Inputs::Str(s) => s,
            }
        }
    }

    #[tokio::test]
    async fn test_heterogeneous_types() -> Result<()> {
        let mut qdrant = TestQdrant::<Inputs>::new();
        let namespace = "test_heterogeneous";

        let _ = qdrant.0.delete_namespace(&namespace).await;
        qdrant.0.create_namespace(&namespace).await?;

        let keys: Vec<Inputs> = vec![
            Inputs::Str("test key 1".to_string()),
            Inputs::Str("test key 2".to_string()),
            Inputs::Ex(Example::new("test key 3".to_string())),
            Inputs::Ex(Example::new("test key 4".to_string())),
            Inputs::Ex(Example::new("test key 5".to_string())),
        ];

        qdrant.0.add_items(&namespace, keys).await?;

        let query = Inputs::Str("test query".to_string());
        let k = 5;
        let result = qdrant.0.knn(&namespace, &query, k).await?;

        assert_eq!(result.len(), 5);

        qdrant
            .0
            .delete_item(&namespace, Inputs::Str("test key 1".to_string()))
            .await?;
        let query = Inputs::Str("test query".to_string());

        let result = qdrant.0.knn(&namespace, &query, k).await?;

        assert_eq!(result.len(), k - 1);

        qdrant.0.delete_namespace(&namespace).await?;

        Ok(())
    }
}
