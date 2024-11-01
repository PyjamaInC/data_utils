// use polars::prelude::*;
// use std::collections::HashMap;
// use std::path::Path;

// mod arrow_stats;
// mod stats;
pub mod dataset;
pub mod date_utils;
pub mod stats;
pub mod transformations;
use arrow::compute::kernels::cmp::gt;
use arrow::error::Result as ArrowResult;
use dataset::Dataset;
use date_utils::{DateConversionOptions, DateFormat, ErrorStrategy};
pub mod lazy;
use lazy::LazyDataset;
use transformations::DateConverter;
// USE EXAMPLE
// use your_crate::{Dataset, DateFormat, DateConversionOptions, ErrorStrategy};

fn main() -> ArrowResult<()> {
    // Builder pattern for configuration
    let file_path = "data/CafeF_HNX_090824.csv";

    let mut dataset = Dataset::builder()
        .with_date_options(DateConversionOptions {
            format: DateFormat::YYYYMMDD,
            strict: false,
            error_strategy: ErrorStrategy::SetNull,
            timezone: None,
        })
        .from_csv(file_path)?;

    // Print first 5 rows
    // println!("Initial data preview:");
    // dataset.print(Some(5));

    // Optional: Transform the date column
    let date_converter = DateConverter::new(DateConversionOptions {
        format: DateFormat::YYYYMMDD,
        strict: false,
        error_strategy: ErrorStrategy::SetNull,
        timezone: None,
    });

    dataset.transform_column("<DTYYYYMMDD>", Box::new(date_converter))?;
    // dataset.print(Some(5));

    // Create a lazy computation plan
    let result = LazyDataset::new(dataset)
        // Select specific columns
        .select(vec![
            "<DTYYYYMMDD>".to_string(),
            "<High>".to_string(),
            "<Low>".to_string(),
        ])
        // Filter rows where close price is greater than open price
        .filter(|batch| {
            let open_price = batch.column(batch.schema().index_of("<High>").unwrap());
            let close_price = batch.column(batch.schema().index_of("<Low>").unwrap());

            // Assuming these are Float64Array
            let open_array = arrow::array::Float64Array::from(open_price.to_data());
            let close_array = arrow::array::Float64Array::from(close_price.to_data());

            gt(&open_array, &close_array).map_err(|e| {
                arrow::error::ArrowError::ComputeError(format!("Error comparing arrays: {}", e))
            })
        })
        // Sort by date in ascending order
        .sort_by(
            vec!["<DTYYYYMMDD>".to_string()],
            vec![true], // true for ascending
        )
        // Limit to first 10 rows
        .limit(10)
        // Execute the plan
        .collect()?;

    println!("\nFiltered and sorted data:");
    result.print(None); // Print all rows since we already limited to 10

    Ok(())
}
