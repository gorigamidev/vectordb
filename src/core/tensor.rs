// src/tensor.rs
use serde::{Deserialize, Serialize};

/// Identificador de tensor (newtype para no confundir con otros u64)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorId(pub u64);

/// Representa la forma (shape) de un tensor.
/// []        -> escalar (rank 0)
/// [3]       -> vector 3D (rank 1)
/// [2, 3]    -> matriz 2x3 (rank 2)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape {
    pub dims: Vec<usize>,
}

impl Shape {
    /// Crea un nuevo shape a partir de una lista de dimensiones
    pub fn new<D: Into<Vec<usize>>>(dims: D) -> Self {
        Self { dims: dims.into() }
    }

    /// Número de dimensiones (rank)
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Número total de elementos
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }
}

/// Tensor denso de f32 con layout row-major
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    pub id: TensorId,
    pub shape: Shape,
    pub data: Vec<f32>,
    #[cfg(feature = "zero-copy")]
    #[serde(skip)] // Don't serialize shared data for now
    pub shared_data: Option<std::sync::Arc<Vec<f32>>>,
}

impl Tensor {
    /// Crea un tensor verificando que data.len() coincide con shape.num_elements()
    pub fn new(id: TensorId, shape: Shape, data: Vec<f32>) -> Result<Self, String> {
        let expected = shape.num_elements();
        if data.len() != expected {
            return Err(format!(
                "Data length {} does not match shape {:?} (expected {})",
                data.len(),
                shape.dims,
                expected
            ));
        }

        Ok(Self {
            id,
            shape,
            data,
            #[cfg(feature = "zero-copy")]
            shared_data: None,
        })
    }

    /// Creates a tensor from shared data (zero-copy)
    #[cfg(feature = "zero-copy")]
    pub fn from_shared(
        id: TensorId,
        shape: Shape,
        data: std::sync::Arc<Vec<f32>>,
    ) -> Result<Self, String> {
        let expected = shape.num_elements();
        if data.len() != expected {
            return Err(format!(
                "Data length {} does not match shape {:?} (expected {})",
                data.len(),
                shape.dims,
                expected
            ));
        }

        Ok(Self {
            id,
            shape,
            data: Vec::new(), // Empty vec when using shared data
            shared_data: Some(data),
        })
    }

    /// Get reference to data slice, handling both owned and shared storage
    pub fn data_ref(&self) -> &[f32] {
        #[cfg(feature = "zero-copy")]
        if let Some(shared) = &self.shared_data {
            return shared;
        }
        &self.data
    }

    /// Get shared reference to data (zero-copy if possible)
    #[cfg(feature = "zero-copy")]
    pub fn share(&self) -> std::sync::Arc<Vec<f32>> {
        if let Some(shared) = &self.shared_data {
            return shared.clone();
        }
        std::sync::Arc::new(self.data.clone())
    }

    /// Get mutable reference to data (copy-on-write if shared)
    #[cfg(feature = "zero-copy")]
    pub fn data_mut(&mut self) -> &mut Vec<f32> {
        if let Some(shared) = self.shared_data.take() {
            // If we have shared data, we must clone it to own it (Copy-On-Write)
            self.data = (*shared).clone();
        }
        &mut self.data
    }

    /// Rank del tensor (0 = escalar, 1 = vector, 2 = matriz...)
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Número de elementos
    pub fn len(&self) -> usize {
        self.data_ref().len()
    }

    pub fn is_empty(&self) -> bool {
        self.data_ref().is_empty()
    }
}
