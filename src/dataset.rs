use crate::date_utils::DateConversionOptions;
// use crate::stats::ColumnStats;
use crate::transformations::ColumnTransform;
use arrow::array::{Date32Array, Float64Array, Int64Array, StringArray};
use arrow::csv::{reader::Format, ReaderBuilder};
use arrow::datatypes::DataType;
use arrow::datatypes::SchemaRef;
use arrow::error::Result as ArrowResult;
use arrow::record_batch::RecordBatch;
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Cell, ContentArrangement, Table};

use chrono::NaiveDate;

use std::fs::File;
use std::io::Seek;
use std::sync::Arc;
pub struct Dataset {
    batches: Vec<RecordBatch>,
    schema: SchemaRef,
}

impl Dataset {
    /// Create a new dataset builder
    pub fn builder() -> DatasetBuilder {
        DatasetBuilder::default()
    }

    /// Get a reference to the internal batches
    pub fn get_batches(&self) -> &Vec<RecordBatch> {
        &self.batches
    }

    /// Create a new Dataset from a vector of RecordBatches
    pub fn new(batches: Vec<RecordBatch>) -> Self {
        let schema = batches
            .first()
            .map(|batch| batch.schema())
            .unwrap_or_else(|| Arc::new(arrow::datatypes::Schema::empty()));

        Self { batches, schema }
    }

    /// Apply a transformation to a column
    pub fn transform_column(
        &mut self,
        column_name: &str,
        transformer: Box<dyn ColumnTransform>,
    ) -> ArrowResult<()> {
        let column_index = self.schema.index_of(column_name)?;

        // println!("Before transformation schema: {:?}", self.schema); // Debug print

        // Transform each batch
        for batch in &mut self.batches {
            let column = batch.column(column_index);
            let transformed = transformer.transform(column)?;

            // Create new schema with updated field type
            let mut fields = self.schema.fields().to_vec();
            fields[column_index] = Arc::new(
                fields[column_index]
                    .as_ref()
                    .clone()
                    .with_data_type(transformed.data_type().clone()),
            );
            self.schema = Arc::new(arrow::datatypes::Schema::new(fields));

            // println!("After transformation schema: {:?}", self.schema); // Debug print

            // Create new batch with transformed column
            let mut columns = batch.columns().to_vec();
            columns[column_index] = transformed.into();

            *batch = RecordBatch::try_new(self.schema.clone(), columns)?;
        }

        Ok(())
    }

    pub fn display(&self, num_rows: Option<usize>) -> String {
        let mut table = Table::new();

        // Set table style
        table
            .load_preset(UTF8_FULL)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_width(200);

        // Create header row that combines column name and type
        let header_cells: Vec<Cell> = self
            .schema
            .fields()
            .iter()
            .map(|field| {
                let type_str = match field.data_type() {
                    DataType::Int64 => "i64",
                    DataType::Float64 => "f64",
                    DataType::Utf8 => "str",
                    DataType::Date32 => "date32",
                    _ => "other",
                };
                Cell::new(format!("{}\n({})", field.name(), type_str))
            })
            .collect();
        table.set_header(header_cells);

        // Rest of the function remains the same...
        let first_batch = match self.batches.first() {
            Some(batch) => batch,
            None => return table.to_string(),
        };

        let rows_to_display = num_rows
            .unwrap_or(first_batch.num_rows())
            .min(first_batch.num_rows());

        // Add data rows
        for row_idx in 0..rows_to_display {
            let row_cells: Vec<Cell> = (0..first_batch.num_columns())
                .map(|col_idx| {
                    let column = first_batch.column(col_idx);
                    let value = match first_batch.schema().field(col_idx).data_type() {
                        DataType::Utf8 => {
                            let array = column.as_any().downcast_ref::<StringArray>().unwrap();
                            array.value(row_idx).to_string()
                        }
                        DataType::Int64 => {
                            let array = column.as_any().downcast_ref::<Int64Array>().unwrap();
                            array.value(row_idx).to_string()
                        }
                        DataType::Float64 => {
                            let array = column.as_any().downcast_ref::<Float64Array>().unwrap();
                            format!("{:.2}", array.value(row_idx))
                        }
                        DataType::Date32 => {
                            let array = column.as_any().downcast_ref::<Date32Array>().unwrap();
                            let days = array.value(row_idx);
                            let date = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap_or_default()
                                + chrono::Duration::days(days as i64);
                            date.format("%Y-%m-%d").to_string()
                        }
                        _ => format!("Unsupported type: {:?}", column.data_type()),
                    };
                    Cell::new(value)
                })
                .collect();
            table.add_row(row_cells);
        }

        // Add summary row if there are more rows
        if rows_to_display < first_batch.num_rows() {
            let remaining = first_batch.num_rows() - rows_to_display;
            let mut summary_row = vec![Cell::new(format!("... and {} more rows", remaining))];
            summary_row.extend(vec![Cell::new("..."); first_batch.num_columns() - 1]);
            table.add_row(summary_row);
        }

        table.to_string()
    }

    // Convenience method to print directly
    pub fn print(&self, num_rows: Option<usize>) {
        println!("{}", self.display(num_rows));
    }

    // Compute statistics for a column
    // pub fn compute_stats(
    //     &self,
    //     column_name: &str,
    //     stats_computer: Box<dyn ColumnStats>,
    // ) -> ArrowResult<Statistics> {
    //     // ... implementation ...
    // }
}

/// Builder pattern for Dataset creation
#[derive(Default)]
pub struct DatasetBuilder {
    schema: Option<SchemaRef>,
    date_options: DateConversionOptions,
    // Add more options as needed
}

impl DatasetBuilder {
    pub fn with_schema(mut self, schema: SchemaRef) -> Self {
        self.schema = Some(schema);
        self
    }

    pub fn with_date_options(mut self, options: DateConversionOptions) -> Self {
        self.date_options = options;
        self
    }

    pub fn from_csv(self, path: &str) -> ArrowResult<Dataset> {
        let mut file = File::open(path)?;
        let format = Format::default().with_header(true).with_delimiter(b',');

        // Use provided schema or infer it
        let schema = if let Some(schema) = self.schema {
            schema
        } else {
            let (inferred_schema, _) = format.infer_schema(&mut file, Some(100))?;
            Arc::new(inferred_schema)
        };

        file.rewind()?;
        let reader = ReaderBuilder::new(schema.clone())
            .with_format(format)
            .build(file)?;

        let batches: Vec<RecordBatch> = reader.into_iter().collect::<ArrowResult<Vec<_>>>()?;

        Ok(Dataset { batches, schema })
    }
}
