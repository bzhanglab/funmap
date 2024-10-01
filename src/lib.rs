use pyo3::prelude::*;

/// funmap_lib
/// Supports the addition of extra features in the format of gene pairs
///
/// Files can reach up to 10 GB, so Rust is used to speed up and optimize the merging process.
///
/// # Step 1: Identify all unique genes
///
/// This step identifies all the unique genes found in both the data and extra-feature files.
///
/// This allows for a universal order to genes that will be referenced in other steps
///
/// Input: All data files and extra feature files
/// Output: .pkl file containing the unique genes found in alphabetical order (A before B)
///
/// # Step 2: Align extra features
///
/// WARN: May need to re-align the regular data files
/// For each extra feature file, re-index the rows according to the order of genes in Step 1.
/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "_lib")] // module name is _lib. (imports as funmap._lib). This is to hide these functions from regular users.
fn funmap_lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
