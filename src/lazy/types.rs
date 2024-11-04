use arrow::array::BooleanArray;
use arrow::error::Result as ArrowResult;
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
/// New type wrappers for statistics results
#[derive(Debug)]
pub struct ColumnMeans(pub HashMap<String, f64>);

#[derive(Debug)]
pub struct ColumnStatistics(pub HashMap<String, (f64, f64)>);

#[derive(Debug)]
pub struct CovarianceMatrix(pub HashMap<(String, String), f64>);

#[derive(Debug)]
pub struct ColumnFullStats {
    pub count: usize,
    pub null_count: usize,
    pub mean: f64,
    pub variance: f64,
    pub median: f64,
    pub max: f64,
    pub min: f64,
}

#[derive(Debug)]
pub struct StatsDescription(pub HashMap<String, ColumnFullStats>);

type FilterFn = dyn Fn(&RecordBatch) -> ArrowResult<BooleanArray> + Send + Sync;

#[derive(Clone)]
pub struct FilterPredicate(pub Arc<FilterFn>);

impl FilterPredicate {
    pub fn new<F>(pred: F) -> Self
    where
        F: Fn(&RecordBatch) -> ArrowResult<BooleanArray> + Send + Sync + 'static,
    {
        Self(Arc::new(pred))
    }

    pub fn apply(&self, batch: &RecordBatch) -> ArrowResult<BooleanArray> {
        (self.0)(batch)
    }
}

// Implement Debug for FilterPredicate
impl fmt::Debug for FilterPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FilterPredicate(<function>)")
    }
}

#[derive(Debug, Clone)]
pub enum Operation {
    Select(Vec<String>),
    Filter(FilterPredicate),
    GroupBy(Vec<String>),
    Sort {
        columns: Vec<String>,
        ascending: Vec<bool>,
    },
    Limit(usize),
}
