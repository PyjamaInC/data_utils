use crate::date_utils::{DateConversionOptions, DateFormat, ErrorStrategy};
use arrow::array::StringArray;
use arrow::array::{Array, Date32Array, Int64Array};
use arrow::datatypes::DataType;
use arrow::error::Result as ArrowResult;
use chrono::{DateTime, NaiveDate};

/// Trait for column transformations
pub trait ColumnTransform {
    fn transform(&self, array: &dyn Array) -> ArrowResult<Box<dyn Array>>;
}

/// Date conversion transformer
pub struct DateConverter {
    options: DateConversionOptions,
}

impl DateConverter {
    pub fn new(options: DateConversionOptions) -> Self {
        Self { options }
    }
}

impl ColumnTransform for DateConverter {
    fn transform(&self, array: &dyn Array) -> ArrowResult<Box<dyn Array>> {
        match array.data_type() {
            DataType::Int64 => {
                let int_array = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    arrow::error::ArrowError::InvalidArgumentError(
                        "Failed to downcast to Int64Array".to_string(),
                    )
                })?;

                let date_array: Date32Array = int_array
                    .iter()
                    .map(|opt_value| {
                        opt_value.and_then(|value| {
                            self.options.format.parse(value).map(|date| {
                                date.signed_duration_since(
                                    NaiveDate::from_ymd_opt(1970, 1, 1).unwrap_or_default(),
                                )
                                .num_days() as i32
                            })
                        })
                    })
                    .collect();

                Ok(Box::new(date_array))
            }
            DataType::Utf8 => {
                let string_array =
                    array
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| {
                            arrow::error::ArrowError::InvalidArgumentError(
                                "Failed to downcast to StringArray".to_string(),
                            )
                        })?;

                let date_array: Date32Array = string_array
                    .iter()
                    .map(|opt_str| {
                        opt_str.and_then(|s| {
                            match &self.options.format {
                                DateFormat::YMD { separator } => {
                                    // Handle "YYYY{sep}MM{sep}DD" format
                                    NaiveDate::parse_from_str(
                                        s,
                                        &format!("%Y{}%m{}%d", separator, separator),
                                    )
                                    .ok()
                                }
                                DateFormat::Custom(fmt) => {
                                    // Handle custom format string
                                    NaiveDate::parse_from_str(s, fmt).ok()
                                }
                                DateFormat::YYYYMMDD => {
                                    // Handle "YYYYMMDD" format without separators
                                    if s.len() != 8 {
                                        return None;
                                    }
                                    let year = s[0..4].parse::<i32>().ok()?;
                                    let month = s[4..6].parse::<u32>().ok()?;
                                    let day = s[6..8].parse::<u32>().ok()?;
                                    NaiveDate::from_ymd_opt(year, month, day)
                                }
                                DateFormat::YYMMDD => {
                                    // Handle "YYMMDD" format
                                    if s.len() != 6 {
                                        return None;
                                    }
                                    let yy = s[0..2].parse::<i32>().ok()?;
                                    let month = s[2..4].parse::<u32>().ok()?;
                                    let day = s[4..6].parse::<u32>().ok()?;
                                    // Assume 20xx for years 00-69, 19xx for 70-99
                                    let year = if yy < 70 { 2000 + yy } else { 1900 + yy };
                                    NaiveDate::from_ymd_opt(year, month, day)
                                }
                                DateFormat::Timestamp => {
                                    // Try to parse as timestamp string
                                    s.parse::<i64>()
                                        .ok()
                                        .and_then(|ts| DateTime::from_timestamp(ts, 0))
                                        .map(|dt| dt.naive_utc().date())
                                }
                            }
                            .map(|date| {
                                date.signed_duration_since(
                                    NaiveDate::from_ymd_opt(1970, 1, 1).unwrap_or_default(),
                                )
                                .num_days() as i32
                            })
                        })
                    })
                    .collect();

                Ok(Box::new(date_array))
            }
            _ => match self.options.error_strategy {
                ErrorStrategy::Strict => {
                    Err(arrow::error::ArrowError::InvalidArgumentError(format!(
                        "Unsupported data type for date conversion: {:?}",
                        array.data_type()
                    )))
                }
                ErrorStrategy::SetNull => {
                    // Return array of nulls
                    Ok(Box::new(Date32Array::from(vec![None; array.len()])))
                }
                ErrorStrategy::UseDefault(default_value) => {
                    // Return array with default value
                    Ok(Box::new(Date32Array::from(vec![
                        Some(default_value);
                        array.len()
                    ])))
                }
            },
        }
    }
}
