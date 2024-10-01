use ahash::{HashSet, HashSetExt};
use pyo3::prelude::*;
use std::{
    fs::File,
    io::{BufRead, BufReader},
};

/// funmap_lib
///
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
///
/// For each extra feature file, re-index the rows according to the order of genes in Step 1.
#[pyfunction]
fn process_files(
    expression_paths: Vec<String>,
    extra_feature_paths: Vec<String>,
) -> PyResult<bool> {
    // Step 1: Identify all unique genes
    // Across both expression and extra_feature_paths
    // Create final unique_gene pkl file

    let mut uniq_gene = HashSet::new();

    // Read expression data where first column is gene information;
    for file_path in expression_paths.iter() {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut has_header = true;
        for line in reader.lines() {
            let line = line?;
            if has_header {
                // skip header
                has_header = false;
                continue;
            }
            let row: Vec<&str> = line.split('\t').collect();
            if row.len() > 1 {
                uniq_gene.insert(row[0].to_string()); // add gene to set
            }
        }
    }

    // Read extra feature data, where first and second column are genes
    for file_path in extra_feature_paths.iter() {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut has_header = true;
        for line in reader.lines() {
            let line = line?;
            if has_header {
                // skip header
                has_header = false;
                continue;
            }
            let row: Vec<&str> = line.split('\t').collect();
            if row.len() > 2 {
                uniq_gene.insert(row[0].to_string()); // add first gene to set
                uniq_gene.insert(row[1].to_string()); // add second gene to set
            }
        }
    }
    Ok(true)
}

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
    m.add_function(wrap_pyfunction!(process_files, m)?)?;
    Ok(())
}
