#[cfg(feature = "zero-copy")]
#[cfg(test)]
mod tests {
    use linal::core::tensor::{Shape, Tensor, TensorId};
    use std::sync::Arc;

    #[test]
    fn test_from_shared() {
        let data = Arc::new(vec![1.0, 2.0, 3.0, 4.0]);
        let shape = Shape::new(vec![2, 2]);
        let id = TensorId(1);

        let tensor =
            Tensor::from_shared(id, shape.clone(), data.clone()).expect("Should create tensor");

        assert_eq!(tensor.len(), 4);
        assert_eq!(tensor.rank(), 2);
        assert_eq!(tensor.data_ref(), &[1.0, 2.0, 3.0, 4.0]);

        // Verify it is actually using the shared memory
        assert!(tensor.shared_data.is_some());
        if let Some(shared) = &tensor.shared_data {
            assert!(Arc::ptr_eq(shared, &data));
        }
    }

    #[test]
    fn test_share_from_owned() {
        let data = vec![1.0, 2.0, 3.0];
        let shape = Shape::new(vec![3]);
        let id = TensorId(2);
        let tensor = Tensor::new(id, shape, data).expect("Should create tensor");

        // calling share on owned tensor should create a new Arc
        let shared = tensor.share();
        assert_eq!(*shared, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_share_from_shared() {
        let data = Arc::new(vec![10.0, 20.0]);
        let shape = Shape::new(vec![2]);
        let id = TensorId(3);
        let tensor = Tensor::from_shared(id, shape, data.clone()).expect("Should create tensor");

        // calling share on already shared tensor should return clone of Arc (cheap)
        let shared_again = tensor.share();
        assert!(Arc::ptr_eq(&data, &shared_again));
    }

    #[test]
    fn test_copy_on_write() {
        let data = Arc::new(vec![1.0, 2.0, 3.0]);
        let shape = Shape::new(vec![3]);
        let id = TensorId(4);
        let mut tensor =
            Tensor::from_shared(id, shape, data.clone()).expect("Should create tensor");

        // Initial state
        assert!(tensor.shared_data.is_some());

        // Mutate data
        let mut_data = tensor.data_mut();
        mut_data[0] = 100.0;

        // Should have detached from shared
        assert!(tensor.shared_data.is_none());
        assert!(!tensor.data.is_empty());
        assert_eq!(tensor.data[0], 100.0);

        // Original Arc data should remain unchanged
        assert_eq!(*data, vec![1.0, 2.0, 3.0]);
    }
}
