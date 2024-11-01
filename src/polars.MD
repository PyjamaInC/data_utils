
// fn calculate_statistics_c<P: AsRef<Path>>(
//     file_path: P,
// ) -> Result<HashMap<String, HashMap<String, f64>>, PolarsError> {
//     let stock = CsvReadOptions::default()
//         .with_has_header(true)
//         .try_into_reader_with_file_path(Some(file_path.as_ref().to_str().unwrap().into()))?
//         .finish()?;

//     let dtype_columns = dtype_cols([
//         DataType::Int64,
//         DataType::Int32,
//         DataType::UInt64,
//         DataType::UInt32,
//         DataType::Float32,
//         DataType::Float64,
//     ])
//     .exclude(["<DTYYYYMMDD>"]);

//     let lazy_stock = stock.lazy().select([dtype_columns]);

//     let result = lazy_stock
//         .select([
//             all().mean().name().suffix("_mean"),
//             all().median().name().suffix("_median"),
//             all().min().name().suffix("_min"),
//             all().max().name().suffix("_max"),
//             all().var(1).name().suffix("_var"),
//             all().std(1).name().suffix("_std"),
//         ])
//         .collect()?;

//     let mut column_stats: HashMap<String, HashMap<String, f64>> = HashMap::new();

//     for column in result.get_columns() {
//         let split_name_vec = column.name().as_str().split('_').collect::<Vec<&str>>();
//         let column_name = split_name_vec[0];
//         let stat_name = split_name_vec[1];

//         column_stats
//             .entry(column_name.to_string())
//             .or_default()
//             .insert(stat_name.to_string(), column.mean().unwrap());
//     }

//     Ok(column_stats)
// }

// fn calculate_statistics_st<P: AsRef<Path>>(
//     file_path: P,
// ) -> Result<HashMap<String, HashMap<String, f64>>, PolarsError> {
//     let stock = CsvReadOptions::default()
//         .with_has_header(true)
//         .try_into_reader_with_file_path(Some(file_path.as_ref().to_str().unwrap().into()))?
//         .finish()?;

//     // Parse the date column
//     let format_stock = stock
//         .lazy()
//         .with_column(
//             col("<DTYYYYMMDD>")
//                 .cast(DataType::String) // Convert i64 to string
//                 .str()
//                 .to_date(StrptimeOptions {
//                     format: Some("%Y%m%d".into()),
//                     exact: true,
//                     ..Default::default()
//                 })
//                 .alias("Date"),
//         )
//         .collect()?;

//     let dtype_columns = dtype_cols([
//         DataType::Int64,
//         DataType::Int32,
//         DataType::UInt64,
//         DataType::UInt32,
//         DataType::Float32,
//         DataType::Float64,
//     ]);

//     let lazy_stock = format_stock.lazy().select([dtype_columns]);

//     let result = lazy_stock
//         .select([
//             all().mean().name().suffix("_mean"),
//             all().median().name().suffix("_median"),
//             all().min().name().suffix("_min"),
//             all().max().name().suffix("_max"),
//             all().var(1).name().suffix("_var"),
//             all().std(1).name().suffix("_std"),
//         ])
//         .collect()?;

//     let mut stat_columns: HashMap<String, HashMap<String, f64>> = HashMap::new();

//     for column in result.get_columns() {
//         let split_name_vec = column.name().as_str().split('_').collect::<Vec<&str>>();
//         let column_name = split_name_vec[0];
//         let stat_name = split_name_vec[1];

//         stat_columns
//             .entry(stat_name.to_string())
//             .or_default()
//             .insert(column_name.to_string(), column.mean().unwrap());
//     }

//     Ok(stat_columns)
// }

// fn main() -> ArrowResult<()> {
//     let file_path = "data/CafeF_HNX_090824.csv";

//     // Read the CSV file into our ArrowDataset
//     let mut dataset = arrow_stats::read_csv_with_arrow(file_path)?;
//     dataset.convert_date_column("<DTYYYYMMDD>")?;
//     // Print a preview of the data (first 5 rows)
//     println!("\nData Preview:");
//     dataset.print_preview(5);

// Compute statistics for specific columns
// Assuming your CSV has numeric columns, replace these with actual column names from your data
// let numeric_columns = ["<Open>", "<High>"];

// println!("\nComputing statistics for numeric columns:");
// for column_name in numeric_columns.iter() {
//     match dataset.compute_stats(column_name) {
//         Ok(stats) => {
//             println!("\nStatistics for {}:", column_name);
//             println!("  Min: {:.2}", stats.min);
//             println!("  Max: {:.2}", stats.max);
//             println!("  Mean: {:.2}", stats.mean);
//             println!("  Median: {:.2}", stats.median);
//         }
//         Err(e) => {
//             println!("Error computing stats for {}: {}", column_name, e);
//         }
//     }
// }

// let stats = calculate_statistics_st(file_path)?;
// Call read_csv_with_arrow and handle the result
// match arrow_stats::read_csv_with_arrow(file_path) {
//     Ok(batches) => {
//         println!("Successfully read {} record batches", batches.len());
//         // You can process the batches here if needed
//     }
//     Err(e) => {
//         eprintln!("Error reading CSV with Arrow: {}", e);
//     }
// }

// for (column, stats) in &stats {
//     println!("Statistics for {}:", column);
//     for (stat, value) in stats {
//         println!("  {}: {}", stat, value);
//     }
//     println!();
// }

// // Add this line to test the arrow array
// arrow_stats::create_arrow_array();

// Ok(())
//}