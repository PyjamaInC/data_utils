use super::types::*;
use crate::dataset::Dataset;
use arrow::array::BooleanArray;
use arrow::error::Result as ArrowResult;
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;

// For basic dataset operations
// implement for LazyDataset in lazy_api.rs
pub trait LazyOperations {
    fn new(dataset: Dataset) -> Self;
    fn select(self, columns: Vec<String>) -> Self;
    fn filter<F>(&self, predicate: F) -> Self
    where
        F: Fn(&RecordBatch) -> ArrowResult<BooleanArray> + Send + Sync + 'static;
    fn sort_by(self, columns: Vec<String>, ascending: Vec<bool>) -> Self;
    fn limit(self, n: usize) -> Self;
    fn collect(self) -> ArrowResult<Dataset>;
}

// For statistical operations
// implement for LazyDataset in statistics.rs
pub trait LazyStatistics {
    fn mean_variance_single_pass(&self, column: &str, ddof: u8) -> ArrowResult<(f64, f64)>;
    fn mean(&self, column: &str) -> ArrowResult<f64>;
    fn counts(&self) -> ArrowResult<HashMap<String, usize>>;
    fn null_counts(&self) -> ArrowResult<HashMap<String, usize>>;
    fn variance(&self, column: &str, ddof: u8) -> ArrowResult<f64>;
    fn means(&self) -> ArrowResult<ColumnMeans>;
    fn variances(&self, ddof: u8) -> ArrowResult<HashMap<String, f64>>;
    fn column_statistics(&self, ddof: u8) -> ArrowResult<ColumnStatistics>;
    fn describe(&self, ddof: u8) -> ArrowResult<StatsDescription>;
    fn calculate_median_max_min(&self, column: &str) -> ArrowResult<(f64, f64, f64)>;
    fn covariance_pair(&self, col1: &str, col2: &str, ddof: u8) -> ArrowResult<f64>;
    fn covariance_matrix(&self, ddof: u8) -> ArrowResult<CovarianceMatrix>;
}

// For internal operations (can be private to the crate)
// implement for LazyDataset in operations.rs
pub(crate) trait LazyOptimizer {
    fn optimize_query_plan(&self) -> Vec<Operation>;
    fn execute_plan(&self, operations: &[Operation]) -> ArrowResult<Dataset>;
}

// For execution of individual operations (can be private to the crate)
// implement for LazyDataset in executor.rs
pub(crate) trait LazyExecutor {
    fn execute_select(
        &self,
        batches: &[RecordBatch],
        columns: &[String],
    ) -> ArrowResult<Vec<RecordBatch>>;

    fn execute_filter(
        &self,
        batches: &[RecordBatch],
        predicate: &FilterPredicate,
    ) -> ArrowResult<Vec<RecordBatch>>;

    fn execute_sort(
        &self,
        batches: &[RecordBatch],
        columns: &[String],
        ascending: &[bool],
    ) -> ArrowResult<Vec<RecordBatch>>;

    fn execute_limit(&self, batches: &[RecordBatch], limit: usize)
        -> ArrowResult<Vec<RecordBatch>>;
}
