use ahash::{AHashMap, AHashSet, HashSet, HashSetExt};
use csv::ReaderBuilder;
use pyo3::{exceptions::PyValueError, prelude::*};
use rusqlite::{params, params_from_iter, Connection, Result};
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
        new(&file_path, &gene_index_map, n as i32, folder_path).expect("Error aligning file");
    }
    // One column of indices, and one column of values. Separated by file
    Ok(true)
}
// Function to parse a string into a float, returning None for invalid or NaN values
fn safe_parse_float(s: &str) -> Option<f64> {
    match s.parse::<f64>() {
        Ok(f) if f.is_nan() => None, // Explicitly keep NaNs as None
        Ok(f) => Some(f),
        Err(_) => None, // If it can't be parsed, treat as None (null)
    }
}
fn new(
    path: &String,
    uniq_gene: &AHashMap<&String, usize>,
    n: i32,
    output_folder: &Path,
) -> PyResult<bool> {
    // Create SQLite connection
    let conn = Connection::open("db.sqlite").unwrap();

    // Open the TSV file
    let file_path = path;
    let mut rdr = ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(file_path)
        .unwrap();

    // Get the headers to determine feature names dynamically
    let headers = rdr.headers().unwrap().clone();

    // Features are all columns after the first two columns
    let feature_names: Vec<&str> = headers
        .iter()
        .skip(2) // Skip the first two columns
        .collect();

    // Dynamically create the SQL table with the appropriate number of features, all as FLOAT
    let feature_columns: Vec<String> = feature_names
        .iter()
        .map(|name| format!("{} FLOAT", name))
        .collect();

    let create_table_query = format!(
        "CREATE TABLE IF NOT EXISTS gene_data (
            index_id TEXT PRIMARY KEY,
            {}
        )",
        feature_columns.join(", ")
    );
    conn.execute(&create_table_query, []).unwrap();

    // Prepare to write features to a separate file (feature names only)
    let feature_file = File::create("features.txt")?;
    let mut feature_writer = BufWriter::new(feature_file);

    // Write feature names to the file
    for feature in &feature_names {
        writeln!(feature_writer, "{}", feature)?;
    }

    // Read each record from TSV and insert it into the database
    for result in rdr.records() {
        let record = result.unwrap();
        let column1 = &record[0];
        let column2 = &record[1];
        let index1 = uniq_gene.get(&column1.to_string());
        let index2 = uniq_gene.get(&column2.to_string());
        if let (Some(index1), Some(index2)) = (index1, index2) {
            let (i, j) = if index1 <= index2 {
                (*index1 as i32, *index2 as i32)
            } else {
                (*index2 as i32, *index1 as i32)
            };
            let index_id = (i * n - i * (i - 1) / 2 + (j - i)) as usize;

            // Extract feature values, converting them to Option<f64> to handle nulls/NaNs
            let feature_values: Vec<Option<f64>> = record
                .iter()
                .skip(2) // Skip the first two columns
                .map(safe_parse_float) // Handle nulls and invalid values
                .collect();

            // Insert into SQLite database using dynamic query
            let insert_query = format!(
                "INSERT OR REPLACE INTO gene_data (index_id, {})
            VALUES (?1, {})",
                feature_names.join(", "),
                feature_values
                    .iter()
                    .enumerate()
                    .map(|(i, _)| format!("?{}", i + 2))
                    .collect::<Vec<String>>()
                    .join(", ")
            );

            // Collect the parameters for the query, using None for null/NaN values
            let mut params_vec: Vec<&(dyn rusqlite::ToSql + Sync)> = vec![&index_id];
            for val in &feature_values {
                match val {
                    Some(v) => params_vec.push(v),
                    None => params_vec.push(&rusqlite::types::Null),
                }
            }

            conn.execute(&insert_query, params_from_iter(params_vec.iter()))
                .unwrap();
        }
    }
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
            let index1 = uniq_gene.get(&row[0].to_string());
            let index2 = uniq_gene.get(&row[1].to_string());
            if let (Some(index1), Some(index2)) = (index1, index2) {
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
