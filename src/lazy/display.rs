use super::types::*;
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Cell, ContentArrangement, Table};
use std::fmt;

// Move all Display implementations here
// Reference lines 30-125 from original lazy.rs
// Implement Display for ColumnFullStatistics
impl fmt::Display for ColumnFullStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut table = Table::new();

        table
            .load_preset(UTF8_FULL)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_width(120);

        table.set_header(vec![
            Cell::new("Column"),
            Cell::new("Mean"),
            Cell::new("Variance"),
            Cell::new("Median"),
            Cell::new("Max"),
            Cell::new("Min"),
        ]);

        let mut entries: Vec<_> = self.0.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));

        for (column, stats) in entries {
            table.add_row(vec![
                Cell::new(column),
                Cell::new(format!("{:.6}", stats.mean)),
                Cell::new(format!("{:.6}", stats.variance)),
                Cell::new(format!("{:.6}", stats.median)),
                Cell::new(format!("{:.6}", stats.max)),
                Cell::new(format!("{:.6}", stats.min)),
            ]);
        }

        write!(f, "{}", table)
    }
}

// Implement Display for our newtype wrapper instead of directly for HashMap
impl fmt::Display for ColumnMeans {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut table = Table::new();

        table
            .load_preset(UTF8_FULL)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_width(100);

        table.set_header(vec![Cell::new("Column"), Cell::new("Value")]);

        // Sort entries by column name
        let mut entries: Vec<_> = self.0.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));

        for (column, value) in entries {
            table.add_row(vec![Cell::new(column), Cell::new(format!("{:.6}", value))]);
        }

        write!(f, "{}", table)
    }
}

impl fmt::Display for ColumnStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut table = Table::new();

        table
            .load_preset(UTF8_FULL)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_width(120);

        table.set_header(vec![
            Cell::new("Column"),
            Cell::new("Mean"),
            Cell::new("Variance"),
        ]);

        let mut entries: Vec<_> = self.0.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));

        for (column, (mean, variance)) in entries {
            table.add_row(vec![
                Cell::new(column),
                Cell::new(format!("{:.6}", mean)),
                Cell::new(format!("{:.6}", variance)),
            ]);
        }

        write!(f, "{}", table)
    }
}
