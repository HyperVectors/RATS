use csv::Reader;
use std::io::Error;

/// Loads a CSV file from ../../data/{dataset}/{dataset}.csv
/// Returns a Vec of Vec<f64> for features and a Vec<String> for labels.
pub fn load_dataset(dataset_name: &str) -> Result<(Vec<Vec<f64>>, Vec<String>), Error> {
    let file_path = format!("../../data/{0}/{0}.csv", dataset_name);
    let mut rdr = Reader::from_path(&file_path)?;

    let mut features = Vec::new();
    let mut labels = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let mut row = Vec::new();
        for i in 0..record.len() - 1 {
            row.push(record[i].parse::<f64>().unwrap_or(f64::NAN));
        }
        features.push(row);
        labels.push(record[record.len() - 1].to_string());
    }
    Ok((features, labels))
}
