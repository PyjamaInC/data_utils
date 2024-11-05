use crate::lazy::prelude::LazyDataset;
use arrow::array::ArrayRef;
use std::sync::Arc;

#[derive(Clone)]
pub struct Series {
    name: String,
    data: ArrayRef,
    lazy_dataset: Option<Arc<LazyDataset>>, // Reference back to parent
}

impl Series {
    pub fn from_lazy_dataset(lazy: &LazyDataset, column: &str) -> Result<Self> {
        // Create series from lazy dataset
        let data = lazy.collect_column(column)?;

        Ok(Series {
            name: column.to_string(),
            data,
            lazy_dataset: Some(Arc::new(lazy.clone())),
        })
    }

    // Method to access parent dataset if needed
    pub fn parent_dataset(&self) -> Option<Arc<LazyDataset>> {
        self.lazy_dataset.clone()
    }
}
