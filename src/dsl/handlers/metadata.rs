use crate::dsl::{DslError, DslOutput};
use crate::engine::TensorDb;

/// Handle SET DATASET <name> METADATA <key> = <value>
pub fn handle_set_metadata(
    db: &mut TensorDb,
    line: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    // Syntax: SET DATASET users METADATA version = "2"
    let rest = line.strip_prefix("SET DATASET ").unwrap().trim();

    // Split by " METADATA "
    let parts: Vec<&str> = rest.splitn(2, " METADATA ").collect();
    if parts.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected: SET DATASET <name> METADATA <key> = <value>".to_string(),
        });
    }

    let dataset_name = parts[0].trim();
    let kv_part = parts[1].trim();

    // Split by "="
    let kv: Vec<&str> = kv_part.splitn(2, '=').collect();
    if kv.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected: <key> = <value> after METADATA".to_string(),
        });
    }

    let key = kv[0].trim().to_string();
    let value = kv[1].trim().trim_matches('"').to_string();

    // Update metadata in DB
    db.set_dataset_metadata(dataset_name, key.clone(), value.clone())
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;

    Ok(DslOutput::Message(format!(
        "Updated metadata for dataset '{}': {} = {}",
        dataset_name, key, value
    )))
}
