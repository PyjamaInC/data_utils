mod display;
pub mod lazy_api;
mod operations;
mod statistics;
pub mod traits;
pub mod types;

// New modules for separated implementations
mod executor;
mod optimizer;

pub use types::{
    ColumnFullStats, ColumnMeans, ColumnStatistics, FilterPredicate, Operation, StatsDescription,
};

pub mod prelude {
    pub use super::lazy_api::LazyDataset;
    pub use super::traits::*; // Re-export all traits
}
