use crate::lazy::prelude::LazyDataset;
use crate::series::series_api::Series;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
/// Flyweight cache for Series instances
pub struct SeriesCache {
    cache: HashMap<String, Arc<Series>>,
}

/// Facade for managing Series operations and caching
pub struct SeriesManager {
    cache: Arc<Mutex<SeriesCache>>,
    lazy_dataset: Arc<LazyDataset>,
}

/// Singleton implementation for SeriesManager
impl SeriesManager {
    /// Get or create the global instance
    pub fn instance(lazy_dataset: LazyDataset) -> Arc<Self> {
        static INSTANCE: std::sync::OnceLock<Arc<Mutex<Option<Arc<SeriesManager>>>>> =
            std::sync::OnceLock::new();

        let manager = INSTANCE.get_or_init(|| Arc::new(Mutex::new(None)));

        let mut guard = manager.lock().unwrap();
        if guard.is_none() {
            *guard = Some(Arc::new(SeriesManager {
                cache: Arc::new(Mutex::new(SeriesCache {
                    cache: HashMap::new(),
                })),
                lazy_dataset: Arc::new(lazy_dataset),
            }));
        }

        guard.as_ref().unwrap().clone()
    }
}

/// Main Series interface
impl SeriesManager {
    pub fn get_series(&self, column: &str, cache_enabled: bool) -> Result<Arc<Series>> {
        if cache_enabled {
            let cache = self.cache.lock().unwrap();
            if let Some(series) = cache.cache.get(column) {
                return Ok(series.clone());
            }
        }

        // Create new series
        let series = Arc::new(Series::from_lazy_dataset(&self.lazy_dataset, column)?);

        if cache_enabled {
            let mut cache = self.cache.lock().unwrap();
            cache.cache.insert(column.to_string(), series.clone());
        }

        Ok(series)
    }
}
