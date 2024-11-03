use super::lazy_api::LazyDataset;
use super::types::*;
use arrow::array::{Array, Float64Array};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use arrow::error::Result as ArrowResult;
use std::arch::aarch64::*;
use std::collections::HashMap;

impl LazyDataset {
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
