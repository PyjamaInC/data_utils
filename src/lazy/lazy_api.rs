use super::types::*;
use crate::dataset::Dataset;
use crate::lazy::traits::{LazyOperations, LazyOptimizer};
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

impl LazyOperations for LazyDataset {
    /// Create a new LazyDataset from a Dataset
    fn new(dataset: Dataset) -> Self {
        Self {
            source: Arc::new(dataset),
            operations: Vec::new(),
            projected_schema: None,
        }
    }

    /// Select specific columns
    /// This operation is pushed down to minimize data reading
    fn select(mut self, columns: Vec<String>) -> Self {
        self.operations.push(Operation::Select(columns));
        // Invalidate cached schema
        self.projected_schema = None;
        self
    }

    /// Filter rows based on a predicate
    /// The predicate is a function that takes a RecordBatch and returns a boolean array
    fn filter<F>(&self, predicate: F) -> Self
    where
        F: Fn(&RecordBatch) -> ArrowResult<BooleanArray> + Send + Sync + 'static,
    {
        let mut new_dataset = Self {
            source: Arc::clone(&self.source),
            operations: self.operations.clone(),
            projected_schema: self.projected_schema.clone(),
        };
        new_dataset
            .operations
            .push(Operation::Filter(FilterPredicate::new(predicate)));
        new_dataset
    }

    /// Sort by columns
    fn sort_by(mut self, columns: Vec<String>, ascending: Vec<bool>) -> Self {
        self.operations.push(Operation::Sort { columns, ascending });
        self
    }

    /// Limit the number of rows
    fn limit(mut self, n: usize) -> Self {
        self.operations.push(Operation::Limit(n));
        self
    }

    /// Execute the query plan and materialize the results
    fn collect(self) -> ArrowResult<Dataset> {
        // Optimize the query plan
        let optimized_ops = self.optimize_query_plan();

        // Execute the optimized plan
        self.execute_plan(&optimized_ops)
    }
}
