// use arrow::error::Result as ArrowResult;
use chrono::DateTime;
use chrono::NaiveDate;
#[derive(Debug, Clone)]
pub enum DateFormat {
    YYYYMMDD,
    YYMMDD,
    YMD { separator: char },
    Custom(String),
    Timestamp,
}

impl DateFormat {
    pub fn parse(&self, value: i64) -> Option<NaiveDate> {
        match self {
            DateFormat::YYYYMMDD => {
                let year = (value / 10000) as i32;
                let month = ((value % 10000) / 100) as u32;
                let day = (value % 100) as u32;
                NaiveDate::from_ymd_opt(year, month, day)
            }
            DateFormat::YYMMDD => {
                let yy = (value / 10000) as i32;
                let month = ((value % 10000) / 100) as u32;
                let day = (value % 100) as u32;
                // Assume 20xx for years 00-69, 19xx for 70-99
                let year = if yy < 70 { 2000 + yy } else { 1900 + yy };
                NaiveDate::from_ymd_opt(year, month, day)
            }
            DateFormat::YMD { separator: _ } => {
                // For pre-formatted strings, this would be handled differently
                // in a separate string parsing function
                None
            }
            DateFormat::Custom(_) => {
                // For pre-formatted strings, this would be handled differently
                // in a separate string parsing function
                None
            }
            DateFormat::Timestamp => {
                DateTime::from_timestamp(value, 0).map(|dt| dt.naive_utc().date())
            }
        }
    }
}

// Configuration struct for date conversion
#[derive(Debug, Clone)]
pub struct DateConversionOptions {
    pub format: DateFormat,
    pub strict: bool,
    pub error_strategy: ErrorStrategy,
    pub timezone: Option<String>,
}

#[derive(Debug, Clone)]
pub enum ErrorStrategy {
    Strict,
    SetNull,
    UseDefault(i32), // days since epoch
}

impl Default for DateConversionOptions {
    fn default() -> Self {
        Self {
            format: DateFormat::YYYYMMDD,
            strict: true,
            error_strategy: ErrorStrategy::Strict,
            timezone: None,
        }
    }
}
