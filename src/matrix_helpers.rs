use nalgebra::DMatrix;

pub(crate) fn vecs_to_dyn_matrix(vecvec: Vec<Vec<f64>>) -> Option<DMatrix<f64>> {
	if vecvec.len() == 0 {
		return Some(DMatrix::zeros(0, 0));
	} else if vecvec[0].len() == 0 {
		return Some(DMatrix::zeros(vecvec.len(), 0));
	}

	let cols = vecvec[0].len();
	if vecvec.iter().any(|vec| vec.len() != cols) {
		return None;
	}
	
	let dm = DMatrix::from_iterator(vecvec.len(), vecvec[0].len(),
		vecvec.into_iter().flatten()
		);
	
	Some(dm)
}