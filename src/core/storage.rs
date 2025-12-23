use std::fs;
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;
use arrow::array::{ArrayRef, Float32Array, Int64Array, StringArray, BooleanArray};
use arrow::datatypes::{Schema as ArrowSchema, Field as ArrowField, DataType};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use crate::core::dataset::Dataset;
use crate::core::tensor::Tensor;
use crate::core::value::{Value, ValueType};

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(String),
    
    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    
    #[error("Dataset not found: {0}")]
    DatasetNotFound(String),
    
    #[error("Tensor not found: {0}")]
    TensorNotFound(String),
}

/// Storage engine trait for persisting datasets and tensors
pub trait StorageEngine {
    /// Save a dataset to storage
    fn save_dataset(&self, dataset: &Dataset) -> Result<(), StorageError>;
    
    /// Load a dataset from storage
    fn load_dataset(&self, name: &str) -> Result<Dataset, StorageError>;
    
    /// Check if a dataset exists
    fn dataset_exists(&self, name: &str) -> bool;
    
    /// Delete a dataset
    fn delete_dataset(&self, name: &str) -> Result<(), StorageError>;
    
    /// List all datasets
    fn list_datasets(&self) -> Result<Vec<String>, StorageError>;
    
    /// Save a tensor to storage
    fn save_tensor(&self, name: &str, tensor: &Tensor) -> Result<(), StorageError>;
    
    /// Load a tensor from storage
    fn load_tensor(&self, name: &str) -> Result<Tensor, StorageError>;
    
    /// Check if a tensor exists
    fn tensor_exists(&self, name: &str) -> bool;
    
    /// Delete a tensor
    fn delete_tensor(&self, name: &str) -> Result<(), StorageError>;
    
    /// List all tensors
    fn list_tensors(&self) -> Result<Vec<String>, StorageError>;
}

/// Parquet-based storage implementation
pub struct ParquetStorage {
    base_path: String,
}

impl ParquetStorage {
    pub fn new(base_path: impl Into<String>) -> Self {
        Self {
            base_path: base_path.into(),
        }
    }
    
    fn dataset_path(&self, name: &str) -> String {
        format!("{}/datasets/{}.parquet", self.base_path, name)
    }
    
    fn metadata_path(&self, name: &str) -> String {
        format!("{}/datasets/{}.meta.json", self.base_path, name)
    }
    
    fn tensor_path(&self, name: &str) -> String {
        format!("{}/tensors/{}.json", self.base_path, name)
    }
    
    fn ensure_directories(&self) -> Result<(), StorageError> {
        let datasets_dir = format!("{}/datasets", self.base_path);
        let tensors_dir = format!("{}/tensors", self.base_path);
        fs::create_dir_all(datasets_dir)?;
        fs::create_dir_all(tensors_dir)?;
        Ok(())
    }
    
    /// Convert Dataset to Arrow RecordBatch
    fn dataset_to_record_batch(&self, dataset: &Dataset) -> Result<RecordBatch, StorageError> {
        // Build Arrow schema from dataset schema
        let arrow_fields: Vec<ArrowField> = dataset.schema.fields.iter()
            .map(|f| {
                let data_type = match &f.value_type {
                    ValueType::Int => DataType::Int64,
                    ValueType::Float => DataType::Float32,
                    ValueType::String => DataType::Utf8,
                    ValueType::Bool => DataType::Boolean,
                    _ => DataType::Utf8, // Fallback for complex types (serialize as JSON string)
                };
                ArrowField::new(&f.name, data_type, f.nullable)
            })
            .collect();
        
        let arrow_schema = Arc::new(ArrowSchema::new(arrow_fields));
        
        // Convert rows to Arrow arrays
        let mut arrays: Vec<ArrayRef> = Vec::new();
        
        for field in &dataset.schema.fields {
            let column_data: Vec<&Value> = dataset.rows.iter()
                .map(|row| {
                    row.values.iter()
                        .zip(&row.schema.fields)
                        .find(|(_, f)| f.name == field.name)
                        .map(|(v, _)| v)
                        .unwrap_or(&Value::Null)
                })
                .collect();
            
            let array: ArrayRef = match &field.value_type {
                ValueType::Int => {
                    let values: Vec<Option<i64>> = column_data.iter()
                        .map(|v| match v {
                            Value::Int(i) => Some(*i),
                            Value::Null => None,
                            _ => None,
                        })
                        .collect();
                    Arc::new(Int64Array::from(values))
                }
                ValueType::Float => {
                    let values: Vec<Option<f32>> = column_data.iter()
                        .map(|v| match v {
                            Value::Float(f) => Some(*f),
                            Value::Int(i) => Some(*i as f32),
                            Value::Null => None,
                            _ => None,
                        })
                        .collect();
                    Arc::new(Float32Array::from(values))
                }
                ValueType::String => {
                    let values: Vec<Option<&str>> = column_data.iter()
                        .map(|v| match v {
                            Value::String(s) => Some(s.as_str()),
                            Value::Null => None,
                            _ => None,
                        })
                        .collect();
                    Arc::new(StringArray::from(values))
                }
                ValueType::Bool => {
                    let values: Vec<Option<bool>> = column_data.iter()
                        .map(|v| match v {
                            Value::Bool(b) => Some(*b),
                            Value::Null => None,
                            _ => None,
                        })
                        .collect();
                    Arc::new(BooleanArray::from(values))
                }
                _ => {
                    // For complex types (Vector, Matrix), serialize as JSON strings
                    let values: Vec<Option<String>> = column_data.iter()
                        .map(|v| match v {
                            Value::Null => None,
                            v => Some(serde_json::to_string(v).unwrap_or_else(|_| "null".to_string())),
                        })
                        .collect();
                    Arc::new(StringArray::from(values))
                }
            };
            
            arrays.push(array);
        }
        
        RecordBatch::try_new(arrow_schema, arrays)
            .map_err(|e| StorageError::Arrow(e))
    }
}

impl StorageEngine for ParquetStorage {
    fn save_dataset(&self, dataset: &Dataset) -> Result<(), StorageError> {
        self.ensure_directories()?;
        
        let dataset_name = dataset.metadata.name.as_ref()
            .ok_or_else(|| StorageError::Serialization("Dataset must have a name".to_string()))?;
        
        // Convert to RecordBatch
        let record_batch = self.dataset_to_record_batch(dataset)?;
        
        // Write to Parquet file
        let data_path = self.dataset_path(dataset_name);
        let file = fs::File::create(&data_path)?;
        let props = WriterProperties::builder().build();
        let mut writer = ArrowWriter::try_new(file, record_batch.schema(), Some(props))?;
        writer.write(&record_batch)?;
        writer.close()?;
        
        // Save metadata as JSON
        let meta_path = self.metadata_path(dataset_name);
        let metadata_json = serde_json::to_string_pretty(&dataset.metadata)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        fs::write(&meta_path, metadata_json)?;
        
        Ok(())
    }
    
    fn load_dataset(&self, _name: &str) -> Result<Dataset, StorageError> {
        // TODO: Implement Parquet to Dataset conversion
        Err(StorageError::Serialization("Load not yet implemented".to_string()))
    }
    
    fn dataset_exists(&self, name: &str) -> bool {
        Path::new(&self.dataset_path(name)).exists()
    }
    
    fn delete_dataset(&self, name: &str) -> Result<(), StorageError> {
        let data_path = self.dataset_path(name);
        let meta_path = self.metadata_path(name);
        
        if Path::new(&data_path).exists() {
            fs::remove_file(&data_path)?;
        }
        
        if Path::new(&meta_path).exists() {
            fs::remove_file(&meta_path)?;
        }
        
        Ok(())
    }
    
    fn list_datasets(&self) -> Result<Vec<String>, StorageError> {
        self.ensure_directories()?;
        
        let datasets_dir = format!("{}/datasets", self.base_path);
        let mut datasets = Vec::new();
        
        for entry in fs::read_dir(&datasets_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("parquet") {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    datasets.push(name.to_string());
                }
            }
        }
        
        Ok(datasets)
    }
    
    fn save_tensor(&self, name: &str, tensor: &Tensor) -> Result<(), StorageError> {
        self.ensure_directories()?;
        
        let tensor_path = self.tensor_path(name);
        let tensor_json = serde_json::to_string_pretty(tensor)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        fs::write(&tensor_path, tensor_json)?;
        
        Ok(())
    }
    
    fn load_tensor(&self, name: &str) -> Result<Tensor, StorageError> {
        let tensor_path = self.tensor_path(name);
        
        if !Path::new(&tensor_path).exists() {
            return Err(StorageError::TensorNotFound(name.to_string()));
        }
        
        let tensor_json = fs::read_to_string(&tensor_path)?;
        let tensor: Tensor = serde_json::from_str(&tensor_json)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        
        Ok(tensor)
    }
    
    fn tensor_exists(&self, name: &str) -> bool {
        Path::new(&self.tensor_path(name)).exists()
    }
    
    fn delete_tensor(&self, name: &str) -> Result<(), StorageError> {
        let tensor_path = self.tensor_path(name);
        
        if Path::new(&tensor_path).exists() {
            fs::remove_file(&tensor_path)?;
        }
        
        Ok(())
    }
    
    fn list_tensors(&self) -> Result<Vec<String>, StorageError> {
        self.ensure_directories()?;
        
        let tensors_dir = format!("{}/tensors", self.base_path);
        let mut tensors = Vec::new();
        
        for entry in fs::read_dir(&tensors_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    tensors.push(name.to_string());
                }
            }
        }
        
        Ok(tensors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parquet_storage_basic() {
        let temp_dir = "/tmp/linal_test_storage";
        let storage = ParquetStorage::new(temp_dir);
        
        // Clean up before test
        let _ = fs::remove_dir_all(temp_dir);
        
        // Test directory creation
        storage.ensure_directories().unwrap();
        assert!(Path::new(&format!("{}/datasets", temp_dir)).exists());
        assert!(Path::new(&format!("{}/tensors", temp_dir)).exists());
        
        // Clean up
        let _ = fs::remove_dir_all(temp_dir);
    }
}
