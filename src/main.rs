// use polars::prelude::*;
// use std::collections::HashMap;
// use std::path::Path;

// mod arrow_stats;
// mod stats;
pub mod dataset;
pub mod date_utils;
pub mod stats;
pub mod transformations;
// use arrow::compute::kernels::cmp::gt;
// use crate::lazy::traits::LazyStatistics;
use arrow::error::Result as ArrowResult;
use dataset::Dataset;
use date_utils::{DateConversionOptions, DateFormat, ErrorStrategy};
use lazy::prelude::*; // Import everything commonly needed
use transformations::DateConverter;
pub mod lazy;
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

    let lazy_dataset = LazyDataset::new(dataset);
    // let aav_data = lazy_dataset
    //     .filter(|batch| {
    //         let ticker_array = batch
    //             .column_by_name("<Ticker>")
    //             .expect("Ticker column not found")
    //             .as_any()
    //             .downcast_ref::<arrow::array::StringArray>()
    //             .expect("Column is not a string array");

    //         Ok(arrow::array::BooleanArray::from_iter(
    //             ticker_array.iter().map(|x| Some(x == Some("AAV"))),
    //         ))
    //     })
    //     .collect()?;
    // aav_data.print(Some(5));
    // println!("mean of open is: {}", lazy_dataset.mean("<Open>")?);
    // println!(
    //     "std sample of open is: {}",
    //     lazy_dataset.variance("<Open>", 1)?
    // );

    // let means = lazy_dataset.means()?;
    // println!("Column Means:");
    // println!("{}", means);

    let stats = lazy_dataset.describe(1)?;
    println!("\nColumn Statistics (Mean and Variance):");
    println!("{}", stats);
    // let cov_matrix = lazy_dataset.covariance_matrix(1)?;
    // println!("{}", cov_matrix);

    Ok(())
}
