use arrow::array::Int64Array;
use arrow::array::{Array, Date32Array, Float64Array, StringArray};
use arrow::compute;
// use arrow::compute::cast;
use arrow::compute::kernels::aggregate::{max, min, sum};
use arrow::csv::reader::Format;
use arrow::csv::ReaderBuilder;
use arrow::datatypes::DataType;
use arrow::error::Result as ArrowResult;
use arrow::record_batch::RecordBatch;

use chrono::NaiveDate;

use std::fs::File;
use std::io::Seek;
use std::sync::Arc;

// New struct to hold the batches and metadata
pub struct ArrowDataset {
    batches: Vec<RecordBatch>,
    row_count: usize,
    schema: Arc<arrow::datatypes::Schema>,
}

#[derive(Debug)]
pub struct ColumnStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
}

impl ArrowDataset {
    pub fn new(batches: Vec<RecordBatch>) -> Self {
        let row_count = batches.iter().map(|b| b.num_rows()).sum();
        let schema = batches[0].schema().clone();
        Self {
            batches,
            row_count,
            schema,
        }
    }

    pub fn compute_stats(&self, column_name: &str) -> ArrowResult<ColumnStats> {
        let column_index = self.schema.index_of(column_name).map_err(|_| {
            arrow::error::ArrowError::InvalidArgumentError(format!(
                "Column {} not found",
                column_name
            ))
        })?;

        // First, concatenate arrays from all batches for the specified column
        let arrays: Vec<&dyn Array> = self
            .batches
            .iter()
            .map(|batch| batch.column(column_index).as_ref())
            .collect();

        // Concatenate the arrays
        let concatenated = compute::concat(&arrays)?;

        // Compute statistics based on data type
        match concatenated.data_type() {
            DataType::Int64 => {
                let array = concatenated.as_any().downcast_ref::<Int64Array>().unwrap();
                let min = min(array).unwrap();
                let max = max(array).unwrap();

                // Calculate mean manually using sum
                let sum_val = sum(array).unwrap();
                let mean = sum_val as f64 / array.len() as f64;

                // For median, we need to sort the values
                let mut values: Vec<i64> = array.values().to_vec();
                values.sort_unstable();
                let len = values.len();
                let median = if len % 2 == 0 {
                    (values[len / 2 - 1] + values[len / 2]) as f64 / 2.0
                } else {
                    values[len / 2] as f64
                };

                Ok(ColumnStats {
                    min: min as f64,
                    max: max as f64,
                    mean,
                    median,
                })
            }
            DataType::Float64 => {
                let array = concatenated
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap();
                let min = compute::min(array).unwrap();
                let max = compute::max(array).unwrap();

                // Calculate mean manually using sum
                let sum_val = sum(array).unwrap();
                let mean = sum_val / array.len() as f64;

                // For median, we need to sort the values
                let mut values: Vec<f64> = array.values().to_vec();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let len = values.len();
                let median = if len % 2 == 0 {
                    (values[len / 2 - 1] + values[len / 2]) / 2.0
                } else {
                    values[len / 2]
                };

                Ok(ColumnStats {
                    min,
                    max,
                    mean,
                    median,
                })
            }
            _ => Err(arrow::error::ArrowError::InvalidArgumentError(
                "Only numeric columns (Int64, Float64) are supported".to_string(),
            )),
        }
    }

    pub fn convert_date_column(&mut self, column_name: &str) -> ArrowResult<()> {
        let column_index = self.schema.index_of(column_name)?;

        // Process each batch
        for batch in &mut self.batches {
            let int_column = batch.column(column_index);
            if let Some(int_array) = int_column.as_any().downcast_ref::<Int64Array>() {
                // Convert YYYYMMDD to days since epoch
                let date_array: Date32Array = int_array
                    .iter()
                    .map(|opt_value| {
                        opt_value.map(|value| {
                            let year = (value / 10000) as i32;
                            let month = ((value % 10000) / 100) as u32;
                            let day = (value % 100) as u32;

                            // Convert to NaiveDate then to days since epoch
                            NaiveDate::from_ymd_opt(year, month, day)
                                .map(|date| {
                                    date.signed_duration_since(
                                        NaiveDate::from_ymd_opt(1970, 1, 1).unwrap_or_default(),
                                    )
                                    .num_days() as i32
                                })
                                .unwrap_or(0)
                        })
                    })
                    .collect();

                // Update schema
                // Create a new vector from the existing fields
                let fields: Vec<Arc<arrow::datatypes::Field>> = self
                    .schema
                    .fields()
                    .iter()
                    .enumerate()
                    .map(|(i, field)| {
                        if i == column_index {
                            // Create new field for the date column
                            Arc::new(arrow::datatypes::Field::new(
                                column_name,
                                arrow::datatypes::DataType::Date32,
                                true,
                            ))
                        } else {
                            // Keep existing fields
                            field.clone()
                        }
                    })
                    .collect();
                let new_schema = Arc::new(arrow::datatypes::Schema::new(fields));

                // Create new batch with converted column
                let mut columns: Vec<Arc<dyn Array>> = batch.columns().to_vec();
                columns[column_index] = Arc::new(date_array);

                *batch = RecordBatch::try_new(new_schema.clone(), columns)?;
            }
        }

        // Update schema in the dataset
        self.schema = self.batches[0].schema();
        Ok(())
    }

    pub fn print_preview(&self, num_rows: usize) {
        if let Some(first_batch) = self.batches.first() {
            println!("Schema: {:#?}", first_batch.schema());
            println!("First batch row count: {}", first_batch.num_rows());

            for i in 0..std::cmp::min(num_rows, first_batch.num_rows()) {
                print!("Row {}: ", i);
                for j in 0..first_batch.num_columns() {
                    let column = first_batch.column(j);
                    let value = match first_batch.schema().field(j).data_type() {
                        DataType::Utf8 => {
                            let array = column.as_any().downcast_ref::<StringArray>().unwrap();
                            array.value(i).to_string()
                        }
                        DataType::Int64 => {
                            let array = column.as_any().downcast_ref::<Int64Array>().unwrap();
                            array.value(i).to_string()
                        }
                        DataType::Float64 => {
                            let array = column.as_any().downcast_ref::<Float64Array>().unwrap();
                            array.value(i).to_string()
                        }
                        DataType::Date32 => {
                            let array = column.as_any().downcast_ref::<Date32Array>().unwrap();
                            let days = array.value(i);
                            let date = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap_or_default()
                                + chrono::Duration::days(days as i64);
                            // Format as YYYYMMDD to match original format
                            date.format("%Y-%m-%d").to_string()
                        }
                        _ => format!("Unsupported type: {:?}", column.data_type()),
                    };

                    print!("{}: {} | ", first_batch.schema().field(j).name(), value);
                }
                println!();
            }
        }
    }
}

// Refactored read function
pub fn read_csv_with_arrow(file_path: &str) -> ArrowResult<ArrowDataset> {
    let mut file = File::open(file_path).unwrap();
    let format = Format::default().with_header(true).with_delimiter(b',');
    let (schema, _) = format.infer_schema(&mut file, Some(100))?;

    file.rewind().unwrap();
    let reader = ReaderBuilder::new(Arc::new(schema))
        .with_format(format)
        .build(file)?;

    let batches: Vec<RecordBatch> = reader.into_iter().collect::<ArrowResult<Vec<_>>>()?;
    Ok(ArrowDataset::new(batches))
}
