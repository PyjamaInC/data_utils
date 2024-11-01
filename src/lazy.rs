use crate::dataset::Dataset;
use arrow::array::{Array, ArrayRef, BooleanArray, UInt32Array, UInt64Array};
use arrow::compute::{self, lexsort_to_indices, take, SortColumn, SortOptions}; // Add this import
use arrow::datatypes::{Field, SchemaRef};
use arrow::error::Result as ArrowResult;
use arrow::record_batch::RecordBatch;
use std::fmt;
use std::sync::Arc;

type FilterFn = dyn Fn(&RecordBatch) -> ArrowResult<BooleanArray> + Send + Sync;

#[derive(Clone)]
pub struct FilterPredicate(Arc<FilterFn>);

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

/// Represents a logical operation in our query plan
#[derive(Debug, Clone)]
pub enum Operation {
    /// Select specific columns
    Select(Vec<String>),
    /// Filter rows based on a condition
    Filter(FilterPredicate),
    /// Group by specific columns
    GroupBy(Vec<String>),
    /// Sort by columns with optional ascending/descending
    Sort {
        columns: Vec<String>,
        ascending: Vec<bool>,
    },
    /// Limit the number of rows
    Limit(usize),
}

/// Represents a lazy computation on a dataset
/// Instead of executing operations immediately, it builds a query plan
/// that can be optimized before execution
pub struct LazyDataset {
    /// The source dataset
    source: Arc<Dataset>,
    /// The sequence of operations to be performed
    operations: Vec<Operation>,
    /// Cached schema that would result from executing all operations
    projected_schema: Option<SchemaRef>,
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

    /// Optimize the query plan before execution
    fn optimize_query_plan(&self) -> Vec<Operation> {
        let mut optimized = Vec::new();
        let mut select_columns: Option<Vec<String>> = None;
        let mut filters = Vec::new();
        let mut sort_op: Option<Operation> = None;
        let mut limit_op: Option<Operation> = None;

        // Collect and reorganize operations
        for op in &self.operations {
            match op {
                // Combine all SELECT operations
                Operation::Select(cols) => {
                    select_columns = Some(match select_columns {
                        Some(existing) => {
                            // Keep only columns that exist in both selections
                            existing
                                .into_iter()
                                .filter(|c| cols.contains(c))
                                .collect::<Vec<_>>()
                        }
                        None => cols.clone(),
                    });
                }
                // Collect all filters to potentially combine them
                Operation::Filter(pred) => filters.push(pred.clone()),
                // Keep only the last sort operation
                Operation::Sort { columns, ascending } => {
                    sort_op = Some(Operation::Sort {
                        columns: columns.clone(),
                        ascending: ascending.clone(),
                    });
                }
                // Keep only the smallest limit
                Operation::Limit(n) => {
                    limit_op = Some(match limit_op {
                        Some(Operation::Limit(existing)) => Operation::Limit(existing.min(*n)),
                        None => Operation::Limit(*n),
                        _ => unreachable!(),
                    });
                }
                _ => optimized.push((*op).clone()),
            }
        }

        // Reconstruct optimized plan in the most efficient order:
        // 1. First SELECT (projection pushdown)
        if let Some(cols) = select_columns {
            optimized.insert(0, Operation::Select(cols));
        }

        // 2. Then FILTERs (predicate pushdown)
        if !filters.is_empty() {
            // Combine multiple filters into one if possible
            optimized.extend(filters.into_iter().map(Operation::Filter));
        }

        // 3. Then SORT (if any)
        if let Some(sort) = sort_op {
            optimized.push(sort);
        }

        // 4. Finally LIMIT (if any)
        if let Some(limit) = limit_op {
            optimized.push(limit);
        }

        optimized
    }

    /// Execute the optimized query plan
    /// Execute the optimized query plan
    fn execute_plan(&self, operations: &[Operation]) -> ArrowResult<Dataset> {
        let mut current_batches = self.source.get_batches().clone();

        for op in operations {
            match op {
                Operation::Select(columns) => {
                    current_batches = self.execute_select(&current_batches, columns)?;
                }
                Operation::Filter(predicate) => {
                    current_batches = self.execute_filter(&current_batches, predicate)?;
                }
                Operation::Sort { columns, ascending } => {
                    current_batches = self.execute_sort(&current_batches, columns, ascending)?;
                }
                Operation::Limit(n) => {
                    current_batches = self.execute_limit(&current_batches, *n)?;
                }
                _ => {} // Handle other operations
            }
        }

        Ok(Dataset::new(current_batches))
    }

    /// Execute SELECT operation
    fn execute_select(
        &self,
        batches: &[RecordBatch],
        columns: &[String],
    ) -> ArrowResult<Vec<RecordBatch>> {
        let mut result = Vec::with_capacity(batches.len());

        for batch in batches {
            let projected_columns: Vec<Arc<dyn Array>> = columns
                .iter()
                .map(|col| {
                    let idx = batch.schema().index_of(col)?;
                    Ok(batch.column(idx).clone())
                })
                .collect::<ArrowResult<_>>()?;

            let projected_schema = Arc::new(arrow::datatypes::Schema::new(
                columns
                    .iter()
                    .map(|col| {
                        let field = batch.schema().field_with_name(col).unwrap().clone();
                        field
                    })
                    .collect::<Vec<Field>>(),
            ));

            result.push(RecordBatch::try_new(projected_schema, projected_columns)?);
        }

        Ok(result)
    }

    /// Execute FILTER operation
    fn execute_filter(
        &self,
        batches: &[RecordBatch],
        predicate: &FilterPredicate,
    ) -> ArrowResult<Vec<RecordBatch>> {
        let mut result = Vec::with_capacity(batches.len());

        for batch in batches {
            let mask = predicate.apply(batch)?;
            let filtered_batch = compute::filter_record_batch(batch, &mask)?;
            if filtered_batch.num_rows() > 0 {
                result.push(filtered_batch);
            }
        }

        Ok(result)
    }

    /// Execute SORT operation
    fn execute_sort(
        &self,
        batches: &[RecordBatch],
        columns: &[String],
        ascending: &[bool],
    ) -> ArrowResult<Vec<RecordBatch>> {
        if batches.is_empty() {
            return Ok(Vec::new());
        }

        // First, concatenate all batches into one
        let mut concatenated_columns: Vec<Vec<ArrayRef>> =
            vec![Vec::new(); batches[0].num_columns()];
        for batch in batches {
            for (i, column) in batch.columns().iter().enumerate() {
                concatenated_columns[i].push(column.clone());
            }
        }

        let concatenated_arrays: Vec<ArrayRef> = concatenated_columns
            .into_iter()
            .map(|column_chunks| {
                // Convert Vec<ArrayRef> to Vec<&dyn Array>
                let arrays: Vec<&dyn Array> = column_chunks
                    .iter()
                    .map(|array| array.as_ref() as &dyn Array)
                    .collect();
                compute::concat(&arrays)
            })
            .collect::<ArrowResult<_>>()?;

        // Create the concatenated batch
        let concatenated_batch = RecordBatch::try_new(batches[0].schema(), concatenated_arrays)?;

        // Get the sort key columns and convert them to SortColumn
        let sort_columns: Vec<SortColumn> = columns
            .iter()
            .zip(ascending.iter())
            .map(|(col, &asc)| {
                let idx = concatenated_batch.schema().index_of(col)?;
                Ok(SortColumn {
                    values: concatenated_batch.column(idx).clone(),
                    options: Some(SortOptions {
                        descending: !asc,
                        nulls_first: true,
                    }),
                })
            })
            .collect::<ArrowResult<_>>()?;

        // Generate sort indices
        let indices = lexsort_to_indices(&sort_columns, None)?;

        // Apply the sort indices to all columns
        let sorted_columns: Vec<ArrayRef> = concatenated_batch
            .columns()
            .iter()
            .map(|col| {
                let taken = take(col, &indices, None)?;
                Ok(Arc::new(taken) as ArrayRef)
            })
            .collect::<ArrowResult<_>>()?;

        // Create the sorted batch
        let sorted_batch = RecordBatch::try_new(batches[0].schema(), sorted_columns)?;

        // Split back into reasonably sized batches
        let target_batch_size = 1024 * 1024; // 1 million rows per batch
        let mut result = Vec::new();
        let total_rows = sorted_batch.num_rows();
        let mut start_idx = 0;

        while start_idx < total_rows {
            let end_idx = (start_idx + target_batch_size).min(total_rows);
            let indices: UInt32Array = (start_idx..end_idx).map(|i| i as u32).collect();

            let batch_columns: Vec<ArrayRef> = sorted_batch
                .columns()
                .iter()
                .map(|col| {
                    let taken = take(col, &indices, None)?;
                    Ok(Arc::new(taken) as ArrayRef)
                })
                .collect::<ArrowResult<_>>()?;

            result.push(RecordBatch::try_new(batches[0].schema(), batch_columns)?);
            start_idx = end_idx;
        }

        Ok(result)
    }

    /// Execute LIMIT operation
    fn execute_limit(
        &self,
        batches: &[RecordBatch],
        limit: usize,
    ) -> ArrowResult<Vec<RecordBatch>> {
        let mut result = Vec::new();
        let mut remaining = limit;

        for batch in batches {
            if remaining == 0 {
                break;
            }

            if batch.num_rows() <= remaining {
                result.push(batch.clone());
                remaining -= batch.num_rows();
            } else {
                // Create an index array from 0 to remaining-1
                let indices: UInt64Array = (0..remaining as u64).collect();

                // Need to slice the batch using take operation
                let columns: Vec<Arc<dyn Array>> = batch
                    .columns()
                    .iter()
                    .map(|col| {
                        let taken = take(col, &indices, None)?;
                        Ok(taken)
                    })
                    .collect::<ArrowResult<_>>()?;

                result.push(RecordBatch::try_new(batch.schema(), columns)?);
                break;
            }
        }

        Ok(result)
    }
}
