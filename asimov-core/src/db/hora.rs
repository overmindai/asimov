use std::{collections::HashMap, sync::Arc};

use parking_lot::Mutex;

use async_trait::async_trait;
use hora::{core::ann_index::ANNIndex, index::hnsw_idx::HNSWIndex};

use crate::{
    error::{AsimovError, Result},
    io::{Embeddable, Input},
    models::Embed,
};

use super::{namespace::Namespace, space::VectorSpace};

use typed_builder::TypedBuilder;

struct HoraCollection {
    collection: HNSWIndex<f32, u64>,
    store: HashMap<u64, Vec<f32>>,
}

impl HoraCollection {
    fn new(dim: usize) -> Self {
        Self {
            collection: HNSWIndex::new(
                dim,
                &hora::index::hnsw_params::HNSWParams::<f32>::default(),
            ),
            store: HashMap::new(),
        }
    }

    fn add_batch(&mut self, keys: impl IntoIterator<Item = (u64, Vec<f32>)>) -> Result<()> {
        for (id, embedding) in keys {
            self.collection.add(&embedding, id).unwrap();
            self.store.insert(id, embedding); // store the vector
        }
        self.collection
            .build(hora::core::metrics::Metric::CosineSimilarity)
            .map_err(|e| AsimovError::Hora(format!("Failed to build collection: {}", e)))?;
        Ok(())
    }

    fn delete_batch(&mut self, ids: Vec<u64>) -> Result<()> {
        let not_found: Vec<u64> = ids
            .iter()
            .filter(|id| !self.store.contains_key(id))
            .copied()
            .collect();

        if !not_found.is_empty() {
            return Err(AsimovError::KeyNotFound(format!(
                "Could not find {not_found:?}"
            )));
        }

        for id in ids {
            self.store.remove(&id);
        }

        self.collection.clear();

        for (&id, embedding) in self.store.iter() {
            let _ = self.collection.add(embedding, id);
        }

        self.collection
            .build(hora::core::metrics::Metric::CosineSimilarity)
            .unwrap();

        Ok(())
    }

    fn delete(&mut self, id: u64) -> Result<()> {
        self.delete_batch(vec![id])
    }

    fn search(&self, embedding: &[f32], k: usize) -> Vec<u64> {
        self.collection.search(embedding, k).into_iter().collect()
    }
}

#[derive(TypedBuilder)]
pub struct HoraDb<E: Embed, I: Embeddable> {
    llm: E,
    #[builder(default, setter(skip))]
    collections: Arc<Mutex<HashMap<Namespace, HoraCollection>>>,
    #[builder(default, setter(skip))]
    store: HashMap<u64, I>,
    #[builder(default, setter(skip))]
    marker: std::marker::PhantomData<I>,
}

impl<E, I> HoraDb<E, I>
where
    E: Embed,
    I: Embeddable,
{
    pub fn new(llm: E) -> Self {
        Self {
            llm,
            collections: Arc::new(Mutex::new(HashMap::new())),
            store: HashMap::new(),
            marker: std::marker::PhantomData,
        }
    }
}

fn uuid_to_u64() -> u64 {
    let uuid = uuid::Uuid::new_v4();
    let bytes = uuid.as_bytes();
    let mut byte_array = [0u8; 8];
    byte_array.copy_from_slice(&bytes[..8]);
    u64::from_ne_bytes(byte_array)
}

#[async_trait]
impl<E, I> VectorSpace for HoraDb<E, I>
where
    E: Embed,
    I: Embeddable + Clone + 'static,
{
    type Item = I;

    async fn namespace_exists(&mut self, namespace: &str) -> Result<bool> {
        let ns = namespace
            .try_into()
            .map_err(|_| AsimovError::InvalidNamespace)?;
        Ok(self.collections.lock().contains_key(&ns))
    }

    async fn create_namespace(&mut self, namespace: &str) -> Result<()> {
        let ns = namespace
            .try_into()
            .map_err(|_| AsimovError::InvalidNamespace)?;

        if self.collections.lock().contains_key(&ns) {
            return Err(AsimovError::KeyCollision(namespace.to_string()));
        }

        let collection = HoraCollection::new(E::DIM as usize);

        self.collections.lock().insert(ns, collection);

        Ok(())
    }

    async fn delete_namespace(&mut self, namespace: &str) -> Result<()> {
        let ns = namespace
            .try_into()
            .map_err(|_| AsimovError::InvalidNamespace)?;
        self.collections
            .lock()
            .remove(&ns)
            .ok_or(AsimovError::KeyNotFound(namespace.to_string()))
            .map(|_| ())
    }

    async fn add_items<It>(&mut self, namespace: &str, keys: It) -> Result<()>
    where
        It: IntoIterator<Item = Self::Item> + Send,
        <It as IntoIterator>::IntoIter: Send,
    {
        let ns = namespace
            .try_into()
            .map_err(|_| AsimovError::InvalidNamespace)?;
        if !self.collections.lock().contains_key(&ns) {
            return Err(AsimovError::KeyNotFound(namespace.to_string()));
        }

        let mut points = Vec::new();

        for item in keys {
            let text = item.key().render()?;
            let embedding = self.llm.embed(&text).await?;
            let id = uuid_to_u64();
            self.store.insert(id, item);
            points.push((id, embedding));
        }

        let mut collections = self.collections.lock();

        let collection = collections
            .get_mut(&ns)
            .ok_or(AsimovError::KeyNotFound(namespace.to_string()))?;

        collection.add_batch(points).map_err(|_| {
            AsimovError::Hora("Could not add vectors to the Hora collection.".to_string())
        })?;

        Ok(())
    }

    async fn delete_item(&mut self, namespace: &str, key: Self::Item) -> Result<()> {
        let ns = namespace
            .try_into()
            .map_err(|_| AsimovError::InvalidNamespace)?;
        if !self.collections.lock().contains_key(&ns) {
            return Err(AsimovError::KeyNotFound(namespace.to_string()));
        }

        let text = key.render()?;
        let id = text.hash()?;

        let mut collections = self.collections.lock();

        let collection = collections
            .get_mut(&ns)
            .ok_or(AsimovError::KeyNotFound(namespace.to_string()))?;

        collection.delete(id)?;

        self.store.remove(&id);

        Ok(())
    }

    async fn knn<K: Input>(&self, namespace: &str, query: &K, k: usize) -> Result<Vec<Self::Item>> {
        let query_clone = query.render()?;

        let embedding = self.llm.embed(&query_clone).await?;

        let ns = namespace
            .try_into()
            .map_err(|_| AsimovError::InvalidNamespace)?;

        let collections = self.collections.lock();

        let collection = collections
            .get(&ns)
            .ok_or(AsimovError::KeyNotFound(namespace.to_string()))?;

        let search_results = collection.search(&embedding, k);

        // TODO: this needs to be examined in more detail.
        let items = search_results
            .into_iter()
            .filter_map(|id| {
                self.store
                    .get(&id)
                    .map(|stored_item| (*stored_item).clone())
            })
            .collect();

        Ok(items)
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[cfg(feature = "openai")]
    use crate::models::openai::OpenAiEmbedding;
    use crate::tokenizers::openai::OpenAiTiktoken;

    struct MockEmbed;

    #[async_trait]
    impl Embed for MockEmbed {
        type Tokenizer = OpenAiTiktoken;
        const DIM: u32 = 128;

        async fn embed<I: Input + ?Sized>(&self, _input: &I) -> Result<Vec<f32>> {
            let embedding = vec![0.0; Self::DIM as usize];
            Ok(embedding)
        }
    }

    #[cfg(feature = "openai")]
    #[tokio::test]
    async fn test_hora_db() -> Result<()> {
        let mut db = HoraDb::new(OpenAiEmbedding::default());

        let namespace = "namespace1";

        // Test create_namespace
        db.create_namespace(&namespace).await?;

        assert!(db
            .collections
            .lock()
            .contains_key(&namespace.try_into().unwrap()));

        let keys = vec![
            "test key 1".to_string(),
            "test key 2".to_string(),
            "test key 3".to_string(),
            "test key 4".to_string(),
            "test key 5".to_string(),
        ];

        db.add_items(&namespace, keys).await?;

        // Test knn
        let query = "test query".to_string();
        let k = 3;
        let result = db.knn(&namespace, &query, k).await?;

        assert_eq!(result.len(), k);
        // Test delete_namespace
        db.delete_namespace(&namespace).await?;
        assert!(!db
            .collections
            .lock()
            .contains_key(&namespace.try_into().unwrap()));

        Ok(())
    }
}
