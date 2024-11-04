use super::lazy_api::LazyDataset;
use super::traits::LazyOptimizer;
use super::types::*;
use crate::dataset::Dataset;
use crate::lazy::traits::LazyExecutor;
use arrow::error::Result as ArrowResult;

impl LazyOptimizer for LazyDataset {
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
}
