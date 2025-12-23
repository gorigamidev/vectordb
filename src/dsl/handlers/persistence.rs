use crate::core::storage::{ParquetStorage, StorageEngine};
use crate::dsl::{DslError, DslOutput};
use crate::engine::TensorDb;

/// Handle SAVE command
/// Syntax: SAVE DATASET dataset_name TO "path"
///         SAVE TENSOR tensor_name TO "path"
pub fn handle_save(db: &mut TensorDb, line: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let rest = line.strip_prefix("SAVE ").unwrap().trim();
    
    if rest.starts_with("DATASET ") {
        handle_save_dataset(db, rest, line_no)
    } else if rest.starts_with("TENSOR ") {
        handle_save_tensor(db, rest, line_no)
    } else {
        Err(DslError::Parse {
            line: line_no,
            msg: "Expected 'DATASET' or 'TENSOR' after 'SAVE'".to_string(),
        })
    }
}

fn handle_save_dataset(db: &mut TensorDb, rest: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let rest = rest.strip_prefix("DATASET ").unwrap().trim();
    
    // Find TO keyword
    let parts: Vec<&str> = rest.splitn(2, " TO ").collect();
    if parts.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected 'TO' keyword in SAVE command".to_string(),
        });
    }
    
    let dataset_name = parts[0].trim();
    let path = parts[1].trim().trim_matches('"');
    
    // Get dataset from store using public method
    let dataset = db.get_dataset(dataset_name)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;
    
    // Save using storage engine
    let storage = ParquetStorage::new(path);
    storage.save_dataset(dataset)
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Failed to save dataset: {}", e),
        })?;
    
    Ok(DslOutput::Message(format!(
        "Saved dataset '{}' to '{}'",
        dataset_name, path
    )))
}

fn handle_save_tensor(_db: &mut TensorDb, _rest: &str, line_no: usize) -> Result<DslOutput, DslError> {
    // TODO: Implement once TensorDb has get_tensor_by_name method
    Err(DslError::Parse {
        line: line_no,
        msg: "SAVE TENSOR not yet implemented - requires tensor lookup method in TensorDb".to_string(),
    })
}

/// Handle LOAD command
/// Syntax: LOAD DATASET dataset_name FROM "path"
///         LOAD TENSOR tensor_name FROM "path"
pub fn handle_load(db: &mut TensorDb, line: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let rest = line.strip_prefix("LOAD ").unwrap().trim();
    
    if rest.starts_with("DATASET ") {
        handle_load_dataset(db, rest, line_no)
    } else if rest.starts_with("TENSOR ") {
        handle_load_tensor(db, rest, line_no)
    } else {
        Err(DslError::Parse {
            line: line_no,
            msg: "Expected 'DATASET' or 'TENSOR' after 'LOAD'".to_string(),
        })
    }
}

fn handle_load_dataset(_db: &mut TensorDb, _rest: &str, line_no: usize) -> Result<DslOutput, DslError> {
    // TODO: Implement load dataset functionality
    Err(DslError::Parse {
        line: line_no,
        msg: "LOAD DATASET command not yet implemented".to_string(),
    })
}

fn handle_load_tensor(_db: &mut TensorDb, _rest: &str, line_no: usize) -> Result<DslOutput, DslError> {
    // TODO: Implement once TensorDb has add_tensor_with_name method
    Err(DslError::Parse {
        line: line_no,
        msg: "LOAD TENSOR not yet implemented - requires add_tensor_with_name method in TensorDb".to_string(),
    })
}

/// Handle LIST DATASETS command
/// Syntax: LIST DATASETS FROM "path"
///         LIST TENSORS FROM "path"
pub fn handle_list_datasets(db: &mut TensorDb, line: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let rest = line.strip_prefix("LIST ").unwrap().trim();
    
    if rest.starts_with("DATASETS") {
        handle_list_datasets_impl(db, rest, line_no)
    } else if rest.starts_with("TENSORS") {
        handle_list_tensors_impl(db, rest, line_no)
    } else {
        Err(DslError::Parse {
            line: line_no,
            msg: "Expected 'DATASETS' or 'TENSORS' after 'LIST'".to_string(),
        })
    }
}

fn handle_list_datasets_impl(_db: &mut TensorDb, rest: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let rest = rest.strip_prefix("DATASETS").unwrap().trim();
    
    if !rest.starts_with("FROM ") {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected 'FROM' keyword in LIST DATASETS command".to_string(),
        });
    }
    
    let path = rest.strip_prefix("FROM ").unwrap().trim().trim_matches('"');
    
    let storage = ParquetStorage::new(path);
    let datasets = storage.list_datasets()
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Failed to list datasets: {}", e),
        })?;
    
    let message = if datasets.is_empty() {
        format!("No datasets found in '{}'", path)
    } else {
        format!("Datasets in '{}':\n  - {}", path, datasets.join("\n  - "))
    };
    
    Ok(DslOutput::Message(message))
}

fn handle_list_tensors_impl(_db: &mut TensorDb, rest: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let rest = rest.strip_prefix("TENSORS").unwrap().trim();
    
    if !rest.starts_with("FROM ") {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected 'FROM' keyword in LIST TENSORS command".to_string(),
        });
    }
    
    let path = rest.strip_prefix("FROM ").unwrap().trim().trim_matches('"');
    
    let storage = ParquetStorage::new(path);
    let tensors = storage.list_tensors()
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Failed to list tensors: {}", e),
        })?;
    
    let message = if tensors.is_empty() {
        format!("No tensors found in '{}'", path)
    } else {
        format!("Tensors in '{}':\n  - {}", path, tensors.join("\n  - "))
    };
    
    Ok(DslOutput::Message(message))
}
