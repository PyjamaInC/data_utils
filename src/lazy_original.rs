use crate::dataset::Dataset;
use arrow::array::{Array, ArrayRef, BooleanArray, Float64Array, UInt32Array, UInt64Array};
use arrow::compute::cast;
use arrow::compute::{self, lexsort_to_indices, take, SortColumn, SortOptions}; // Add this import
use arrow::datatypes::DataType;
use arrow::datatypes::{Field, SchemaRef};
use arrow::error::Result as ArrowResult;
use arrow::record_batch::RecordBatch;
use std::arch::aarch64::*;
use std::collections::HashMap;
use std::sync::Arc;

use comfy_table::presets::UTF8_FULL;
use comfy_table::{Cell, ContentArrangement, Table};
use std::fmt;

#[derive(Debug)]
pub struct ColumnFullStats {
    pub mean: f64,
    pub variance: f64,
    pub median: f64,
    pub max: f64,
    pub min: f64,
}

#[derive(Debug)]
pub struct ColumnFullStatistics(pub HashMap<String, ColumnFullStats>);

// Implement Display for ColumnFullStatistics
impl fmt::Display for ColumnFullStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut table = Table::new();

        table
            .load_preset(UTF8_FULL)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_width(120);

        table.set_header(vec![
            Cell::new("Column"),
            Cell::new("Mean"),
            Cell::new("Variance"),
            Cell::new("Median"),
            Cell::new("Max"),
            Cell::new("Min"),
        ]);

        let mut entries: Vec<_> = self.0.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));

        for (column, stats) in entries {
            table.add_row(vec![
                Cell::new(column),
                Cell::new(format!("{:.6}", stats.mean)),
                Cell::new(format!("{:.6}", stats.variance)),
                Cell::new(format!("{:.6}", stats.median)),
                Cell::new(format!("{:.6}", stats.max)),
                Cell::new(format!("{:.6}", stats.min)),
            ]);
        }

        write!(f, "{}", table)
    }
}

// Newtype wrappers for our statistics results
#[derive(Debug)]
pub struct ColumnMeans(pub HashMap<String, f64>);

#[derive(Debug)]
pub struct ColumnStatistics(pub HashMap<String, (f64, f64)>);

// Implement Display for our newtype wrapper instead of directly for HashMap
impl fmt::Display for ColumnMeans {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut table = Table::new();

        table
            .load_preset(UTF8_FULL)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_width(100);

        table.set_header(vec![Cell::new("Column"), Cell::new("Value")]);

        // Sort entries by column name
        let mut entries: Vec<_> = self.0.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));

        for (column, value) in entries {
            table.add_row(vec![Cell::new(column), Cell::new(format!("{:.6}", value))]);
        }

        write!(f, "{}", table)
    }
}

impl fmt::Display for ColumnStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut table = Table::new();

        table
            .load_preset(UTF8_FULL)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_width(120);

        table.set_header(vec![
            Cell::new("Column"),
            Cell::new("Mean"),
            Cell::new("Variance"),
        ]);

        let mut entries: Vec<_> = self.0.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));

        for (column, (mean, variance)) in entries {
            table.add_row(vec![
                Cell::new(column),
                Cell::new(format!("{:.6}", mean)),
                Cell::new(format!("{:.6}", variance)),
            ]);
        }

        write!(f, "{}", table)
    }
}

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

    /// Optimizes the query execution plan by consolidating and reordering operations
    /// for more efficient execution.
    ///
    /// This method processes the sequence of operations in the query plan and applies
    /// various optimization strategies:
    ///
    /// # Optimization Strategies
    ///
    /// ## SELECT Operations
    /// - Combines multiple SELECT operations by computing their intersection
    /// - Ensures column selections are consistent throughout the query
    ///
    /// ```rust
    /// // These queries are optimized to be equivalent:
    /// dataset
    ///     .select(vec!["name", "age", "city"])
    ///     .select(vec!["age", "city"]);
    /// // Optimizes to: SELECT ["age", "city"]
    ///
    /// dataset.select(vec!["age", "city"]);
    /// ```
    ///
    /// ## FILTER Operations
    /// - Collects all filter predicates for potential combination
    /// - Maintains the order of filters for optimal row reduction
    ///
    /// ```rust
    /// // Multiple filters are preserved
    /// dataset
    ///     .filter(|batch| /* age > 25 */)
    ///     .filter(|batch| /* city == "New York" */);
    /// // Both filters will be applied in sequence
    /// ```
    ///
    /// ## SORT Operations
    /// - Keeps only the last sort operation as multiple sorts are redundant
    /// - Preserves the sort columns and their ordering (ascending/descending)
    ///
    /// ```rust
    /// // In this query, only the last sort is kept
    /// dataset
    ///     .sort_by(vec!["age"], vec![true])
    ///     .sort_by(vec!["name"], vec![false]);
    /// // Optimizes to: SORT BY name DESC
    /// ```
    ///
    /// ## LIMIT Operations
    /// - Combines multiple LIMIT operations by taking the minimum value
    /// - Ensures the smallest limit is applied for optimal performance
    ///
    /// ```rust
    /// // These queries are optimized to be equivalent:
    /// dataset
    ///     .limit(100)
    ///     .limit(50);
    /// // Optimizes to: LIMIT 50
    ///
    /// dataset.limit(50);
    /// ```
    ///
    /// # Operation Reordering
    ///
    /// Operations are reordered for optimal execution in the following sequence:
    /// 1. SELECT (projection pushdown)
    /// 2. FILTER (predicate pushdown)
    /// 3. SORT
    /// 4. LIMIT
    ///
    /// This ordering ensures that:
    /// - Data is filtered and projected as early as possible
    /// - Expensive operations (like SORT) work with minimal data
    /// - LIMIT is applied last to ensure correct results
    ///
    /// # Memory Considerations
    ///
    /// The optimization process uses additional memory to:
    /// - Store intermediate operation states
    /// - Track column selections
    /// - Maintain filter predicates
    ///
    /// However, this memory usage is typically negligible compared to the performance
    /// benefits of the optimization.
    ///
    /// # Thread Safety
    ///
    /// The optimization process is thread-safe as it:
    /// - Uses immutable references to the original operations
    /// - Creates new operation instances where needed
    /// - Maintains thread-safety guarantees of the original predicates
    ///
    /// # Examples
    ///
    /// Basic query optimization:
    /// ```rust
    /// let query = dataset
    ///     .select(vec!["name", "age"])
    ///     .filter(|batch| /* age > 25 */)
    ///     .sort_by(vec!["name"], vec![true])
    ///     .limit(10);
    /// ```
    ///
    /// Multiple operations that get optimized:
    /// ```rust
    /// let query = dataset
    ///     .select(vec!["name", "age", "city"])
    ///     .select(vec!["age", "city"])  // Combined with first SELECT
    ///     .filter(|batch| /* age > 25 */)
    ///     .sort_by(vec!["age"], vec![true])
    ///     .sort_by(vec!["name"], vec![false])  // Only last SORT kept
    ///     .limit(100)
    ///     .limit(50);  // Minimized to LIMIT 50
    /// ```
    ///
    /// # Returns
    ///
    /// Returns a new Vec<Operation> containing the optimized sequence of operations.
    /// The optimized plan maintains the same logical results as the original while
    /// potentially improving execution performance.
    ///
    /// # Implementation Details
    ///
    /// The optimization process uses several internal data structures:
    /// - `select_columns`: Option<Vec<String>> for tracking column selections
    /// - `filters`: Vec<FilterPredicate> for collecting filter operations
    /// - `sort_op`: Option<Operation> for the last sort operation
    /// - `limit_op`: Option<Operation> for the minimum limit
    ///
    /// Each operation type is handled specifically:
    /// ```rust
    /// // SELECT handling
    /// select_columns = Some(match select_columns {
    ///     Some(existing) => existing.into_iter()
    ///         .filter(|c| cols.contains(c))
    ///         .collect(),
    ///     None => cols.clone(),
    /// });
    ///
    /// // FILTER handling
    /// filters.push(predicate.clone());
    ///
    /// // SORT handling
    /// sort_op = Some(Operation::Sort {
    ///     columns: columns.clone(),
    ///     ascending: ascending.clone(),
    /// });
    ///
    /// // LIMIT handling
    /// limit_op = Some(match limit_op {
    ///     Some(Operation::Limit(existing)) =>
    ///         Operation::Limit(existing.min(*n)),
    ///     None => Operation::Limit(*n),
    ///     _ => unreachable!(),
    /// });
    /// ```
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

    /// Executes a SELECT operation on a set of record batches by projecting specified columns.
    ///
    /// This method performs column projection on each record batch in the input set, creating
    /// new record batches that contain only the specified columns while preserving their order
    /// and data types.
    ///
    /// # Arguments
    ///
    /// * `batches` - A slice of RecordBatch instances to process. Each RecordBatch represents
    ///               a collection of arrays (columns) with the same length and a shared schema.
    ///
    /// * `columns` - A slice of column names to select. These names must exist in the schema
    ///               of the input batches. The order of columns in this slice determines the
    ///               order in the output batches.
    ///
    /// # Returns
    ///
    /// Returns `ArrowResult<Vec<RecordBatch>>`, where:
    /// - On success: A vector of new RecordBatch instances containing only the selected columns
    /// - On failure: An Arrow error (e.g., if a column name doesn't exist in the schema)
    ///
    /// # Memory Efficiency and Data Sharing
    ///
    /// This method is designed to be extremely memory-efficient through the use of Arc (Atomic
    /// Reference Counting). No actual column data is ever copied; instead, only references are
    /// manipulated. Here's how the memory layout works:
    ///
    /// ```text
    /// Initial State:
    /// Original RecordBatch
    /// ├── Schema (Arc)
    /// └── Columns [
    ///     ├── Column1 (Arc) ─────→ [Actual Array Data]
    ///     ├── Column2 (Arc) ─────→ [Actual Array Data]
    ///     └── Column3 (Arc) ─────→ [Actual Array Data]
    /// ]
    ///
    /// After Selection (e.g., selecting Column1 and Column2):
    /// New RecordBatch
    /// ├── New Schema (Arc, only selected fields)
    /// └── New Columns [
    ///     ├── Column1 (Arc) ─┐
    ///     └── Column2 (Arc) ─┤
    ///                        │
    /// Original RecordBatch   │
    /// ├── Schema (Arc)       │
    /// └── Columns [          │
    ///     ├── Column1 (Arc) ←┤
    ///     ├── Column2 (Arc) ←┘
    ///     └── Column3 (Arc) ─────→ [Actual Array Data]
    /// ]
    /// ```
    ///
    /// Key points about memory usage:
    /// 1. Column data is never copied, only Arc references are cloned
    /// 2. Each Arc clone only increments a reference counter
    /// 3. Memory overhead is minimal, consisting only of:
    ///    - New RecordBatch structure (metadata)
    ///    - New Schema structure (metadata)
    ///    - New Arc pointers to existing columns
    ///
    /// # Internal Mechanism
    ///
    /// For each input batch, the method:
    /// 1. Creates projected columns by:
    ///    - Mapping each requested column name to its index in the original schema
    ///    - Cloning the corresponding column arrays (only Arc cloning, no data copying)
    ///    ```rust
    ///    let projected_columns: Vec<Arc<dyn Array>> = columns
    ///        .iter()
    ///        .map(|col| {
    ///            let idx = batch.schema().index_of(col)?;
    ///            Ok(batch.column(idx).clone())  // Only clones the Arc pointer
    ///        })
    ///        .collect::<ArrowResult<_>>()?;
    ///    ```
    ///
    /// 2. Constructs a new schema containing only the selected fields:
    ///    - Preserves field metadata and data types from the original schema
    ///    - Maintains the order specified in the input columns slice
    ///    ```rust
    ///    let projected_schema = Arc::new(Schema::new(
    ///        columns
    ///            .iter()
    ///            .map(|col| batch.schema().field_with_name(col).unwrap().clone())
    ///            .collect()
    ///    ));
    ///    ```
    ///
    /// 3. Creates a new RecordBatch with the projected columns and schema
    ///
    /// # Performance Considerations
    ///
    /// - Pre-allocates the result vector with the exact capacity needed
    /// - Uses efficient column cloning through Arc (reference counting)
    /// - Maintains batch boundaries for consistent memory usage
    /// - Zero-copy operation for actual data
    /// - Minimal memory overhead for metadata structures
    ///
    /// # Examples
    ///
    /// ```rust
    /// let batches = vec![
    ///     // RecordBatch with columns: "name", "age", "city"
    ///     // ...
    /// ];
    ///
    /// // Select only "name" and "age" columns
    /// let selected = execute_select(&batches, &["name", "age"])?;
    /// // Result contains batches with only "name" and "age" columns
    /// // Original data is shared, not copied
    /// ```
    ///
    /// # Errors
    ///
    /// This method will return an error if:
    /// - Any requested column name doesn't exist in the input batch schema
    /// - The creation of new RecordBatch instances fails
    /// - Column data types are incompatible with the operation
    ///
    /// # Memory Usage Example
    ///
    /// ```rust
    /// // Original batch with 1 million rows, 3 columns (e.g., 100MB of data)
    /// let original_batch = /* ... */;
    ///
    /// // Selecting 2 columns
    /// let selected = execute_select(&[original_batch], &["col1", "col2"])?;
    ///
    /// // Memory impact:
    /// // - Original data (100MB): stays in place, unchanged
    /// // - New structures: ~few KB for metadata and Arc pointers
    /// // - Total additional memory: negligible (just metadata)
    /// // - Actual column data: shared via Arc, no duplication
    /// ```
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

    /// Executes a sorting operation on a collection of RecordBatches based on specified columns and sort orders.
    /// This method implements a multi-step process to efficiently sort potentially large datasets while
    /// maintaining memory efficiency and Arrow's columnar format advantages.
    ///
    /// # Algorithm Overview
    ///
    /// The sorting process consists of four main phases:
    /// 1. Batch Concatenation
    /// 2. Multi-column Sorting
    /// 3. Data Reordering
    /// 4. Batch Splitting
    ///
    /// # Arguments
    ///
    /// * `batches` - A slice of RecordBatch instances to be sorted. Each batch contains the same schema
    ///               but may have different numbers of rows.
    /// * `columns` - A slice of column names that define the sort keys. The order of columns determines
    ///               the primary, secondary, etc. sort keys.
    /// * `ascending` - A slice of boolean values corresponding to each sort column, where `true` indicates
    ///                ascending order and `false` indicates descending order.
    ///
    /// # Returns
    ///
    /// Returns `ArrowResult<Vec<RecordBatch>>`, containing the sorted data split into reasonably-sized
    /// batches. Each output batch maintains the same schema as the input batches.
    ///
    /// # Memory Management
    ///
    /// The method employs several strategies to manage memory efficiently:
    ///
    /// 1. **Zero-Copy Operations**:
    ///    - Uses Arrow's internal reference counting (Arc) to avoid unnecessary data copying
    ///    - Maintains columnar format throughout the process
    ///    - Leverages Arrow's compute kernels for efficient operations
    ///
    /// 2. **Batch Size Control**:
    ///    - Final output is split into batches of approximately 1 million rows each
    ///    - Prevents excessive memory usage while maintaining processing efficiency
    ///
    /// # Implementation Details
    ///
    /// ## Phase 1: Batch Concatenation
    /// ```text
    /// Input Batches:    Concatenated Result:
    /// Batch1 [A B C]    [A B C]
    /// Batch2 [D E F] => [D E F]
    /// Batch3 [G H I]    [G H I]
    /// ```
    /// - Creates a single large batch for sorting
    /// - Groups corresponding columns across all input batches
    /// - Uses Arrow's concat operation for efficient column merging
    ///
    /// ## Phase 2: Sort Configuration
    /// ```rust
    /// SortColumn {
    ///     values: column_data,
    ///     options: SortOptions {
    ///         descending: !ascending,
    ///         nulls_first: true
    ///     }
    /// }
    /// ```
    /// - Configures sort parameters for each column
    /// - Handles null value positioning
    /// - Supports mixed ascending/descending orders
    ///
    /// ## Phase 3: Sorting and Reordering
    /// 1. Generates sort indices using lexicographical sorting
    /// 2. Applies indices to reorder all columns consistently
    /// 3. Maintains data integrity across all columns
    ///
    /// ## Phase 4: Batch Splitting
    /// ```text
    /// Large Sorted Batch:     Final Output:
    /// [All Sorted Data] => Batch1 [1M rows]
    ///                      Batch2 [1M rows]
    ///                      Batch3 [Remaining rows]
    /// ```
    /// - Splits result into manageable chunks
    /// - Maintains consistent batch sizes (1M rows)
    /// - Preserves original schema
    ///
    /// # Performance Considerations
    ///
    /// 1. **Time Complexity**:
    ///    - Concatenation: O(n) where n is total number of rows
    ///    - Sorting: O(n log n) for the primary sort operation
    ///    - Splitting: O(n) for final batch creation
    ///
    /// 2. **Space Complexity**:
    ///    - Temporary peak memory usage: O(n) for concatenated batch
    ///    - Additional O(n) for sort indices
    ///    - Final output size equals input size
    ///
    /// # Error Handling
    ///
    /// The method returns errors in the following cases:
    /// - Invalid column names in the sort specification
    /// - Memory allocation failures during any phase
    /// - Arrow computation errors during concatenation or sorting
    ///
    /// # Examples
    ///
    /// ```rust
    /// let batches = vec![batch1, batch2, batch3];
    /// let sort_columns = vec!["age".to_string(), "name".to_string()];
    /// let ascending = vec![true, false];  // age ascending, name descending
    ///
    /// let sorted_batches = dataset.execute_sort(&batches, &sort_columns, &ascending)?;
    /// ```
    ///
    /// # Implementation Notes
    ///
    /// 1. **Null Handling**:
    ///    - Nulls are consistently placed first in the sort order
    ///    - This behavior is controlled by `SortOptions::nulls_first`
    ///
    /// 2. **Memory Efficiency**:
    ///    - Uses Arrow's internal memory management
    ///    - Avoids unnecessary copies through reference counting
    ///    - Splits large results into manageable chunks
    ///
    /// 3. **Thread Safety**:
    ///    - Implementation is thread-safe due to immutable input handling
    ///    - Uses Arc for safe memory sharing
    ///
    /// 4. **Schema Preservation**:
    ///    - Maintains input schema throughout all operations
    ///    - Ensures data type consistency
    ///
    /// # Limitations
    ///
    /// 1. Requires enough memory to hold at least:
    ///    - One copy of the concatenated data
    ///    - Sort indices array
    ///    - One output batch (1M rows)
    ///
    /// 2. Does not currently support:
    ///    - Streaming sort operations
    ///    - Custom null ordering per column
    ///    - External sorting for datasets larger than available memory
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

    fn mean_variance_single_pass(&self, column: &str, ddof: u8) -> ArrowResult<(f64, f64)> {
        let batches = self.source.get_batches();
        let mut m = 0.0f64; // Running mean
        let mut s = 0.0f64; // Running sum of squares of differences from the current mean
        let mut count = 0usize;

        for batch in batches {
            let idx = batch.schema().index_of(column)?;
            let array = batch.column(idx);

            // Cast to f64 if needed
            let array = match array.data_type() {
                DataType::Float64 => array.clone(),
                _ => cast(array, &DataType::Float64)?,
            };

            if let Some(float_array) = array.as_any().downcast_ref::<Float64Array>() {
                let values = float_array.values();

                #[cfg(target_arch = "aarch64")]
                unsafe {
                    // Process 2 f64 values at a time using NEON
                    let chunks = values.chunks_exact(2);
                    let remainder = chunks.remainder();

                    for chunk in chunks {
                        if !float_array.is_null(count) && !float_array.is_null(count + 1) {
                            // Load 2 f64 values into a NEON vector
                            let v = vld1q_f64(chunk.as_ptr());

                            // Current mean as vector
                            let mean_vec = vdupq_n_f64(m);

                            // Calculate differences from current mean
                            let diff = vsubq_f64(v, mean_vec);

                            // Store the old mean
                            let old_m = m;

                            // Update mean using both values
                            let count_f64 = count as f64;
                            m += vgetq_lane_f64(diff, 0) / (count_f64 + 1.0);
                            m += vgetq_lane_f64(diff, 1) / (count_f64 + 2.0);

                            // Update sum of squares
                            s += vgetq_lane_f64(diff, 0) * (vgetq_lane_f64(diff, 0) + (old_m - m));
                            s += vgetq_lane_f64(diff, 1) * (vgetq_lane_f64(diff, 1) + (old_m - m));

                            count += 2;
                        }
                    }

                    // Handle remaining elements
                    for &val in remainder {
                        if !float_array.is_null(count) {
                            let old_m = m;
                            count += 1;
                            let diff = val - old_m;
                            m += diff / count as f64;
                            s += diff * (diff + (old_m - m));
                        }
                    }
                }

                #[cfg(not(target_arch = "aarch64"))]
                {
                    // Fallback for non-ARM architectures
                    for (i, &val) in values.iter().enumerate() {
                        if !float_array.is_null(i) {
                            let old_m = m;
                            count += 1;
                            let diff = val - old_m;
                            m += diff / count as f64;
                            s += diff * (diff + (old_m - m));
                        }
                    }
                }
            }
        }

        if count <= ddof as usize {
            Ok((0.0, 0.0))
        } else {
            Ok((m, s / (count - ddof as usize) as f64))
        }
    }

    /// Calculate variance using SIMD-optimized single-pass algorithm
    pub fn variance(&self, column: &str, ddof: u8) -> ArrowResult<f64> {
        Ok(self.mean_variance_single_pass(column, ddof)?.1)
    }

    /// Calculate mean using SIMD-optimized single-pass algorithm
    pub fn mean(&self, column: &str) -> ArrowResult<f64> {
        Ok(self.mean_variance_single_pass(column, 0)?.0)
    }

    pub fn means(&self) -> ArrowResult<ColumnMeans> {
        let schema = self.source.as_ref().get_schema();
        let mut results = HashMap::new();

        for field in schema.fields() {
            let column_name = field.name();

            // Check if the column type is numeric
            match field.data_type() {
                DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float32
                | DataType::Float64 => {
                    // Calculate mean for numeric column
                    match self.mean(column_name) {
                        Ok(mean) => {
                            results.insert(column_name.clone(), mean);
                        }
                        Err(e) => {
                            // Log warning but continue with other columns
                            eprintln!(
                                "Warning: Could not calculate mean for column {}: {}",
                                column_name, e
                            );
                        }
                    }
                }
                _ => {
                    // Skip non-numeric columns
                    eprintln!(
                        "Info: Skipping non-numeric column: {} (type: {:?})",
                        column_name,
                        field.data_type()
                    );
                }
            }
        }

        Ok(ColumnMeans(results))
    }

    /// Calculate variances for all numeric columns in the dataset
    /// Returns a HashMap mapping column names to their variances
    pub fn variances(&self, ddof: u8) -> ArrowResult<HashMap<String, f64>> {
        let schema = self.source.get_schema();
        let mut results = HashMap::new();

        for field in schema.fields() {
            let column_name = field.name();

            // Check if the column type is numeric
            match field.data_type() {
                DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float32
                | DataType::Float64 => {
                    // Calculate variance for numeric column
                    match self.variance(column_name, ddof) {
                        Ok(var) => {
                            results.insert(column_name.clone(), var);
                        }
                        Err(e) => {
                            // Log warning but continue with other columns
                            eprintln!(
                                "Warning: Could not calculate variance for column {}: {}",
                                column_name, e
                            );
                        }
                    }
                }
                _ => {
                    // Skip non-numeric columns
                    eprintln!(
                        "Info: Skipping non-numeric column: {} (type: {:?})",
                        column_name,
                        field.data_type()
                    );
                }
            }
        }

        Ok(results)
    }

    /// Calculate both mean and variance for all numeric columns in a single pass
    /// Returns a HashMap mapping column names to (mean, variance) tuples
    pub fn column_statistics(&self, ddof: u8) -> ArrowResult<ColumnStatistics> {
        let schema = self.source.get_schema();
        let mut results = HashMap::new();

        for field in schema.fields() {
            let column_name = field.name();

            // Check if the column type is numeric
            match field.data_type() {
                DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float32
                | DataType::Float64 => {
                    // Calculate both mean and variance in single pass
                    match self.mean_variance_single_pass(column_name, ddof) {
                        Ok(stats) => {
                            results.insert(column_name.clone(), stats);
                        }
                        Err(e) => {
                            eprintln!(
                                "Warning: Could not calculate statistics for column {}: {}",
                                column_name, e
                            );
                        }
                    }
                }
                _ => {
                    eprintln!(
                        "Info: Skipping non-numeric column: {} (type: {:?})",
                        column_name,
                        field.data_type()
                    );
                }
            }
        }

        Ok(ColumnStatistics(results))
    }

    fn calculate_median_max_min(&self, column: &str) -> ArrowResult<(f64, f64, f64)> {
        let batches = self.source.get_batches();
        let mut all_values = Vec::new();

        for batch in batches {
            let idx = batch.schema().index_of(column)?;
            let array = batch.column(idx);

            // Cast to f64 if needed
            let array = match array.data_type() {
                DataType::Float64 => array.clone(),
                _ => cast(array, &DataType::Float64)?,
            };

            if let Some(float_array) = array.as_any().downcast_ref::<Float64Array>() {
                for i in 0..float_array.len() {
                    if !float_array.is_null(i) {
                        all_values.push(float_array.value(i));
                    }
                }
            }
        }

        if all_values.is_empty() {
            return Ok((0.0, 0.0, 0.0));
        }

        // Calculate median
        all_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = all_values.len() / 2;
        let median = if all_values.len() % 2 == 0 {
            (all_values[mid - 1] + all_values[mid]) / 2.0
        } else {
            all_values[mid]
        };

        // Calculate max and min
        let max = *all_values.last().unwrap();
        let min = *all_values.first().unwrap();

        Ok((median, max, min))
    }

    /// Calculate comprehensive statistics for all numeric columns
    pub fn column_full_statistics(&self, ddof: u8) -> ArrowResult<ColumnFullStatistics> {
        let schema = self.source.get_schema();
        let mut results = HashMap::new();

        for field in schema.fields() {
            let column_name = field.name();

            // Check if the column type is numeric
            match field.data_type() {
                DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float32
                | DataType::Float64 => {
                    // Calculate mean and variance in single pass
                    match self.mean_variance_single_pass(column_name, ddof) {
                        Ok((mean, variance)) => {
                            // Calculate median, max, and min in second pass
                            match self.calculate_median_max_min(column_name) {
                                Ok((median, max, min)) => {
                                    results.insert(
                                        column_name.clone(),
                                        ColumnFullStats {
                                            mean,
                                            variance,
                                            median,
                                            max,
                                            min,
                                        },
                                    );
                                }
                                Err(e) => {
                                    eprintln!(
                                        "Warning: Could not calculate median/max/min for column {}: {}",
                                        column_name, e
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!(
                                "Warning: Could not calculate mean/variance for column {}: {}",
                                column_name, e
                            );
                        }
                    }
                }
                _ => {
                    eprintln!(
                        "Info: Skipping non-numeric column: {} (type: {:?})",
                        column_name,
                        field.data_type()
                    );
                }
            }
        }

        Ok(ColumnFullStatistics(results))
    }
}
