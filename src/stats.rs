use arrow::datatypes::{DataType, SchemaRef};
use std::collections::HashMap;

use arrow::array::Array;
use arrow::error::Result as ArrowResult;

/// Trait for computing column statistics
pub trait ColumnStats {
    fn compute(&self, array: &dyn Array) -> ArrowResult<Statistics>;
}

#[derive(Debug)]
pub struct Statistics {
    pub numeric: Option<NumericStats>,
    pub categorical: Option<CategoricalStats>,
    pub temporal: Option<TemporalStats>,
}

#[derive(Debug)]
pub struct NumericStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    // Add more as needed
}

#[derive(Debug)]
pub struct CategoricalStats {
    pub unique_count: usize,
    pub most_common: String,
    pub most_common_count: usize,
    // Add more as needed
}

#[derive(Debug)]
pub struct TemporalStats {
    pub min_date: i32,
    pub max_date: i32,
    pub range_days: i32,
    // Add more as needed
}

/// Represents the category of data in a column
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnCategory {
    Numeric,     // Int32, Int64, Float32, Float64
    Categorical, // String, Bool
    Temporal,    // Date, Timestamp
    Unsupported, // Any other type
}

/// Maps Arrow DataType to our ColumnCategory
fn categorize_datatype(dtype: &DataType) -> ColumnCategory {
    match dtype {
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float32
        | DataType::Float64 => ColumnCategory::Numeric,

        DataType::Utf8 | DataType::Boolean => ColumnCategory::Categorical,

        DataType::Date32 | DataType::Date64 | DataType::Timestamp(_, _) => ColumnCategory::Temporal,

        _ => ColumnCategory::Unsupported,
    }
}

/// Analyzes a schema and returns a mapping of column names to their categories
pub fn analyze_schema(schema: SchemaRef) -> HashMap<String, ColumnCategory> {
    schema
        .fields()
        .iter()
        .map(|field| (field.name().clone(), categorize_datatype(field.data_type())))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{Field, Schema};

    #[test]
    fn test_categorize_datatype() {
        assert_eq!(
            categorize_datatype(&DataType::Int64),
            ColumnCategory::Numeric
        );
        assert_eq!(
            categorize_datatype(&DataType::Float64),
            ColumnCategory::Numeric
        );
        assert_eq!(
            categorize_datatype(&DataType::Utf8),
            ColumnCategory::Categorical
        );
        assert_eq!(
            categorize_datatype(&DataType::Date32),
            ColumnCategory::Temporal
        );
    }

    #[test]
    fn test_analyze_schema() {
        let schema = Schema::new(vec![
            Field::new("numeric_col", DataType::Float64, false),
            Field::new("string_col", DataType::Utf8, false),
            Field::new("date_col", DataType::Date32, false),
        ]);

        let categories = analyze_schema(Arc::new(schema));

        assert_eq!(
            categories.get("numeric_col"),
            Some(&ColumnCategory::Numeric)
        );
        assert_eq!(
            categories.get("string_col"),
            Some(&ColumnCategory::Categorical)
        );
        assert_eq!(categories.get("date_col"), Some(&ColumnCategory::Temporal));
    }
}
