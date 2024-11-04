use super::lazy_api::LazyDataset;
use super::traits::LazyStatistics;
use super::types::*;
use arrow::array::{Array, Float64Array};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use arrow::error::Result as ArrowResult;
use std::arch::aarch64::*;
use std::collections::HashMap;

impl LazyStatistics for LazyDataset {
    // Implementation of counts() method
    fn counts(&self) -> ArrowResult<HashMap<String, usize>> {
        let schema = self.source.get_schema();
        let mut counts = HashMap::new();

        for field in schema.fields() {
            let column_name = field.name();
            let mut count = 0;

            // Sum up rows across all batches
            for batch in self.source.get_batches() {
                count += batch.column(batch.schema().index_of(column_name)?).len();
            }

            counts.insert(column_name.clone(), count);
        }

        Ok(counts)
    }

    fn null_counts(&self) -> ArrowResult<HashMap<String, usize>> {
        let schema = self.source.get_schema();
        let mut null_counts = HashMap::new();

        for field in schema.fields() {
            let column_name = field.name();
            let mut null_count = 0;

            // Sum up rows across all batches
            for batch in self.source.get_batches() {
                null_count += batch
                    .column(batch.schema().index_of(column_name)?)
                    .null_count();
            }

            null_counts.insert(column_name.clone(), null_count);
        }

        Ok(null_counts)
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
    fn variance(&self, column: &str, ddof: u8) -> ArrowResult<f64> {
        Ok(self.mean_variance_single_pass(column, ddof)?.1)
    }

    /// Calculate mean using SIMD-optimized single-pass algorithm
    fn mean(&self, column: &str) -> ArrowResult<f64> {
        Ok(self.mean_variance_single_pass(column, 0)?.0)
    }

    fn means(&self) -> ArrowResult<ColumnMeans> {
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
    fn variances(&self, ddof: u8) -> ArrowResult<HashMap<String, f64>> {
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
    fn column_statistics(&self, ddof: u8) -> ArrowResult<ColumnStatistics> {
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
    fn describe(&self, ddof: u8) -> ArrowResult<StatsDescription> {
        let schema = self.source.get_schema();
        let mut results = HashMap::new();

        let counts = self.counts()?;
        let null_counts = self.null_counts()?;
        for field in schema.fields() {
            let column_name = field.name();

            // Get count for this column
            let count = counts.get(column_name).copied().unwrap_or(0);
            let null_count = null_counts.get(column_name).copied().unwrap_or(0);
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
                                            count,
                                            null_count,
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

        Ok(StatsDescription(results))
    }

    fn covariance_pair(&self, col1: &str, col2: &str, ddof: u8) -> ArrowResult<f64> {
        let batches = self.source.get_batches();
        let mut mean1 = 0.0f64;
        let mut mean2 = 0.0f64;
        let mut c12 = 0.0f64; // Covariance * (n-1)
        let mut count = 0usize;

        for batch in batches {
            let idx1 = batch.schema().index_of(col1)?;
            let idx2 = batch.schema().index_of(col2)?;
            let array1 = cast(batch.column(idx1), &DataType::Float64)?;
            let array2 = cast(batch.column(idx2), &DataType::Float64)?;

            if let (Some(float_array1), Some(float_array2)) = (
                array1.as_any().downcast_ref::<Float64Array>(),
                array2.as_any().downcast_ref::<Float64Array>(),
            ) {
                let values1 = float_array1.values();
                let values2 = float_array2.values();

                #[cfg(target_arch = "aarch64")]
                unsafe {
                    let chunks1 = values1.chunks_exact(2);
                    let chunks2 = values2.chunks_exact(2);
                    let remainder1 = chunks1.remainder();
                    let remainder2 = chunks2.remainder();

                    for (chunk1, chunk2) in chunks1.zip(chunks2) {
                        if !float_array1.is_null(count)
                            && !float_array1.is_null(count + 1)
                            && !float_array2.is_null(count)
                            && !float_array2.is_null(count + 1)
                        {
                            // Load 2 f64 values into NEON vectors
                            let v1 = vld1q_f64(chunk1.as_ptr());
                            let v2 = vld1q_f64(chunk2.as_ptr());

                            // Current means as vectors
                            let mean1_vec = vdupq_n_f64(mean1);
                            let mean2_vec = vdupq_n_f64(mean2);

                            // Calculate differences from current means
                            let diff1 = vsubq_f64(v1, mean1_vec);
                            let diff2 = vsubq_f64(v2, mean2_vec);

                            // Update means
                            let count_f64 = count as f64;
                            mean1 += vgetq_lane_f64(diff1, 0) / (count_f64 + 1.0);
                            mean1 += vgetq_lane_f64(diff1, 1) / (count_f64 + 2.0);
                            mean2 += vgetq_lane_f64(diff2, 0) / (count_f64 + 1.0);
                            mean2 += vgetq_lane_f64(diff2, 1) / (count_f64 + 2.0);

                            // Update covariance sum
                            c12 += vgetq_lane_f64(diff1, 0) * vgetq_lane_f64(diff2, 0);
                            c12 += vgetq_lane_f64(diff1, 1) * vgetq_lane_f64(diff2, 1);

                            count += 2;
                        }
                    }

                    // Handle remaining elements
                    for (&val1, &val2) in remainder1.iter().zip(remainder2.iter()) {
                        if !float_array1.is_null(count) && !float_array2.is_null(count) {
                            let old_mean1 = mean1;
                            let old_mean2 = mean2;
                            count += 1;
                            let diff1 = val1 - old_mean1;
                            let diff2 = val2 - old_mean2;
                            mean1 += diff1 / count as f64;
                            mean2 += diff2 / count as f64;
                            c12 += diff1 * diff2;
                        }
                    }
                }

                #[cfg(not(target_arch = "aarch64"))]
                {
                    for i in 0..values1.len() {
                        if !float_array1.is_null(i) && !float_array2.is_null(i) {
                            let val1 = values1[i];
                            let val2 = values2[i];
                            count += 1;
                            let diff1 = val1 - mean1;
                            let diff2 = val2 - mean2;
                            mean1 += diff1 / count as f64;
                            mean2 += diff2 / count as f64;
                            c12 += diff1 * diff2;
                        }
                    }
                }
            }
        }

        if count <= ddof as usize {
            Ok(0.0)
        } else {
            Ok(c12 / (count - ddof as usize) as f64)
        }
    }
    /// Calculate the covariance matrix for all numeric columns
    fn covariance_matrix(&self, ddof: u8) -> ArrowResult<CovarianceMatrix> {
        let schema = self.source.get_schema();
        let mut numeric_columns: Vec<String> = Vec::new();
        let mut results = HashMap::new();

        // First, collect all numeric columns
        for field in schema.fields() {
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
                    numeric_columns.push(field.name().clone());
                }
                _ => continue,
            }
        }

        // Calculate covariance for each pair of columns
        for i in 0..numeric_columns.len() {
            for j in i..numeric_columns.len() {
                let col1 = &numeric_columns[i];
                let col2 = &numeric_columns[j];

                match self.covariance_pair(col1, col2, ddof) {
                    Ok(cov) => {
                        // Store both (i,j) and (j,i) as covariance matrix is symmetric
                        results.insert((col1.clone(), col2.clone()), cov);
                        if i != j {
                            results.insert((col2.clone(), col1.clone()), cov);
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "Warning: Could not calculate covariance for columns {} and {}: {}",
                            col1, col2, e
                        );
                    }
                }
            }
        }

        Ok(CovarianceMatrix(results))
    }
}
