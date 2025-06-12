use csv::Reader;
use std::fs::File;
use std::io::Error;
use std::io::Write;

/// Loads a CSV file from ../../data/{dataset}/{dataset}.csv
/// Returns a Vec of Vec<f64> for features and a Vec<String> for labels.
pub fn load_dataset(dataset_name: &str) -> Result<(Vec<Vec<f64>>, Vec<String>), Error> {
    let file_path = format!("../data/{0}/{0}.csv", dataset_name);
    println!("Loading dataset from: {}", file_path);
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

pub fn write_dataset_csv(
    features: &Vec<Vec<f64>>,
    labels: &Vec<String>,
    dataset_name: &str,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let filepath = format!("../data/{}/{}", dataset_name, filename);
    println!("Writing dataset to: {}", filepath);
    let mut file = File::create(filepath)?;

    // header
    if let Some(first_row) = features.first() {
        let mut header: Vec<String> = (0..first_row.len()).map(|i| format!("t_{}", i)).collect();
        header.push("label".to_string());
        writeln!(file, "{}", header.join(","))?;
    }

    // data rows
    for (row, label) in features.iter().zip(labels) {
        let mut line = row
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<String>>()
            .join(",");
        line.push(',');
        line.push_str(label);
        writeln!(file, "{line}")?;
    }
    Ok(())
}
