use async_trait::async_trait;

use crate::{error::Result, io::Embeddable, Input};

#[async_trait]
pub trait VectorSpace
where
    Self::Item: Embeddable,
{
    type Item;

    async fn namespace_exists(&mut self, namespace: &str) -> Result<bool>;

    async fn create_namespace(&mut self, namespace: &str) -> Result<()>;
    async fn delete_namespace(&mut self, namespace: &str) -> Result<()>;

    async fn add_items<It>(&mut self, namespace: &str, items: It) -> Result<()>
    where
        It: IntoIterator<Item = Self::Item> + Send,
        <It as IntoIterator>::IntoIter: Send;

    async fn add_item(&mut self, namespace: &str, item: Self::Item) -> Result<()> {
        self.add_items(namespace, vec![item]).await
    }
    async fn delete_item(&mut self, namespace: &str, item: Self::Item) -> Result<()>;

    async fn knn<I: Input>(&self, namespace: &str, query: &I, k: usize) -> Result<Vec<Self::Item>>;
}
