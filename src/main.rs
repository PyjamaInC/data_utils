// use polars::prelude::*;
// use std::collections::HashMap;
// use std::path::Path;

// mod arrow_stats;
// mod stats;
pub mod dataset;
pub mod date_utils;
pub mod stats;
pub mod transformations;
use arrow::error::Result as ArrowResult;
use dataset::Dataset;
use date_utils::{DateConversionOptions, DateFormat, ErrorStrategy};
pub mod lazy;
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
    println!("Initial data preview:");
    dataset.print(Some(5));

    // Optional: Transform the date column
    let date_converter = DateConverter::new(DateConversionOptions {
        format: DateFormat::YYYYMMDD,
        strict: false,
        error_strategy: ErrorStrategy::SetNull,
        timezone: None,
    });

    dataset.transform_column("<DTYYYYMMDD>", Box::new(date_converter))?;
    dataset.print(Some(5));
    Ok(())
}
