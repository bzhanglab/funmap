use ahash::{AHashMap, AHashSet, HashSet, HashSetExt};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde_pickle::SerOptions;
use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

/// process_files
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
/// For each extra feature file, re-index the rows according to the order of genes in Step 1.
///
/// Then save each column as a separate pkl file
///
/// Input: unique genes from Step 1, extra feature files.
/// Output: One pkl file for each feature across all extra feature files
///
/// Function
/// Output: list of all output feature pkl files
///
/// TODO: Protein-coding gene filtering
#[pyfunction]
#[pyo3(signature = (expression_paths, extra_feature_paths, output_folder, valid_ids=None))]
fn process_files(
    expression_paths: Vec<String>,
    extra_feature_paths: Vec<String>,
    output_folder: String,
    valid_ids: Option<Vec<String>>,
) -> PyResult<bool> {
    // Step 1: Identify all unique genes
    // Across both expression and extra_feature_paths
    // Create final unique_gene pkl file

    let mut uniq_gene = HashSet::new();

    // Read expression data where first column is gene information;
    for file_path in expression_paths.iter() {
        let file = File::open(file_path).expect("Could not read expression data");
        let reader = BufReader::new(file);
        let mut has_header = true;
        for line in reader.lines() {
            let line = line?;
            if has_header {
                // skip header
                has_header = false;
                let row: Vec<&str> = line.split('\t').collect();
                if row.len() < 2 {
                    return Err(PyValueError::new_err(format!(
                        "Expression data file at {} does not have enough columns.",
                        file_path
                    )));
                }
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
        let file = File::open(file_path).expect("Could not read extra feature file");
        let reader = BufReader::new(file);
        let mut has_header = true;
        for line in reader.lines() {
            let line = line?;
            if has_header {
                // skip header
                has_header = false;
                let row: Vec<&str> = line.split('\t').collect();
                if row.len() < 3 {
                    return Err(PyValueError::new_err(format!(
                        "Extra feature file at {} does not have enough columns.",
                        file_path
                    )));
                }
                continue;
            }
            let row: Vec<&str> = line.split('\t').collect();
            if row.len() > 2 {
                uniq_gene.insert(row[0].to_string()); // add first gene to set
                uniq_gene.insert(row[1].to_string()); // add second gene to set
            }
        }
    }

    let mut uniq_gene: Vec<String> = if let Some(valid_ids) = valid_ids {
        let valid_ids = AHashSet::from_iter(valid_ids);
        uniq_gene.union(&valid_ids).cloned().collect()
    } else {
        uniq_gene.iter().cloned().collect()
    };

    // Save to pickle
    // TODO: Look at other file formats

    // Sort genes alphabetically
    uniq_gene.sort();
    let folder_path = Path::new(&output_folder);
    let uniq_gene_file_path = folder_path.join("uniq_gene.pkl");
    let mut w = File::create(uniq_gene_file_path).expect("Could not cread uniq_gene.pkl");
    serde_pickle::to_writer(&mut w, &uniq_gene, SerOptions::default()).unwrap();
    let n = uniq_gene.len();
    // Re-align each file
    // Create a HashMap to store the indices of each string
    let mut gene_index_map: AHashMap<&String, usize> = AHashMap::new();
    for (index, gene) in uniq_gene.iter().enumerate() {
        gene_index_map.insert(gene, index);
    }

    for file_path in extra_feature_paths {
        align_file(&file_path, &gene_index_map, n as i32, folder_path)
            .expect("Error aligning file");
    }
    // One column of indices, and one column of values. Separated by file
    Ok(true)
}

fn align_file(
    path: &String,
    uniq_gene: &AHashMap<&String, usize>,
    n: i32,
    output_folder: &Path,
) -> PyResult<bool> {
    // Read extra feature data, where first and second column are genes
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut has_header = true;
    let mut indices: Vec<usize> = Vec::new();
    let mut feature_count = 0;
    let mut writers = Vec::new();
    let original_file = Path::new(path).file_name().unwrap().to_str().unwrap();
    let index_file_path = output_folder.join(format!("{}.index", original_file));
    let f = File::create(index_file_path)?;
    let bf = BufWriter::new(f);
    writers.push(bf);

    for line in reader.lines() {
        let line = line?;
        if has_header {
            // skip header
            has_header = false;
            let row: Vec<&str> = line.split('\t').collect();
            if row.len() < 3 {
                return Err(PyValueError::new_err(format!(
                    "Extra feature file at {} does not have enough columns.",
                    path
                )));
            }
            feature_count = row.len() - 2;
            for i in 0..feature_count {
                let file_path = output_folder.join(format!("{}.col", row[i + 2]));
                let f = File::create(file_path)?;
                let bf = BufWriter::new(f);
                writers.push(bf);
            }
            continue;
        }
        let row: Vec<&str> = line.split('\t').collect();
        if row.len() > 2 {
            let index1 = uniq_gene.get(&row[0].to_string()).unwrap();
            let index2 = uniq_gene.get(&row[1].to_string()).unwrap();
            let (i, j) = if index1 <= index2 {
                (*index1 as i32, *index2 as i32)
            } else {
                (*index2 as i32, *index1 as i32)
            };
            let new_index = (i * n - i * (i - 1) / 2 + (j - i)) as usize;
            indices.push(new_index);
            writers[0].write_all(new_index.to_string().as_bytes())?;
            writers[0].write_all(b"\n")?;
            for i in 0..feature_count {
                writers[i + 1].write_all(row[i + 2].as_bytes())?;
                writers[i + 1].write_all(b"\n")?;
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
