pub trait LazyOperations {
    fn select(self, columns: Vec<String>) -> Self;
    fn filter<F>(self, predicate: F) -> Self
    where
        F: Fn(&RecordBatch) -> ArrowResult<BooleanArray> + Send + Sync + 'static;
    fn sort_by(self, columns: Vec<String>, ascending: Vec<bool>) -> Self;
    fn limit(self, n: usize) -> Self;
}

pub trait LazyStatistics {
    fn mean(&self, column: &str) -> ArrowResult<f64>;
    fn variance(&self, column: &str, ddof: u8) -> ArrowResult<f64>;
    fn means(&self) -> ArrowResult<ColumnMeans>;
    // ... other statistical methods
}
