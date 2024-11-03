use super::types::*;
use crate::dataset::Dataset;
use arrow::array::BooleanArray;
use arrow::datatypes::SchemaRef;
use arrow::error::Result as ArrowResult;
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

pub struct LazyDataset {
    pub(super) source: Arc<Dataset>,
    pub(super) operations: Vec<Operation>,
    pub(super) projected_schema: Option<SchemaRef>,
}

impl LazyDataset {
    /// Create a new LazyDataset from a Dataset
    pub fn new(dataset: Dataset) -> Self {
        Self {
            source: Arc::new(dataset),
            operations: Vec::new(),
            projected_schema: None,
        }
    }

    /// Select specific columns
    /// This operation is pushed down to minimize data reading
    pub fn select(mut self, columns: Vec<String>) -> Self {
        self.operations.push(Operation::Select(columns));
        // Invalidate cached schema
        self.projected_schema = None;
        self
    }

    /// Filter rows based on a predicate
    /// The predicate is a function that takes a RecordBatch and returns a boolean array
    pub fn filter<F>(mut self, predicate: F) -> Self
    where
        F: Fn(&RecordBatch) -> ArrowResult<BooleanArray> + Send + Sync + 'static,
    {
        self.operations
            .push(Operation::Filter(FilterPredicate::new(predicate)));
        self
    }

    /// Sort by columns
    pub fn sort_by(mut self, columns: Vec<String>, ascending: Vec<bool>) -> Self {
        self.operations.push(Operation::Sort { columns, ascending });
        self
    }

    /// Limit the number of rows
    pub fn limit(mut self, n: usize) -> Self {
        self.operations.push(Operation::Limit(n));
        self
    }

    /// Execute the query plan and materialize the results
    pub fn collect(self) -> ArrowResult<Dataset> {
        // Optimize the query plan
        let optimized_ops = self.optimize_query_plan();

        // Execute the optimized plan
        self.execute_plan(&optimized_ops)
    }
}
