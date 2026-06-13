// Column-oriented storage of the Abalone dataset.
// Each field is a Vec holding all values for that feature across every sample.
// We store by column rather than by row because ML operations (mean, std, gradient)
// work on entire features at once — iterating a single Vec<f64> is much faster
// than extracting a field from each row struct.
pub struct AbaloneDataset {
    pub sex: Vec<String>,           // categorical: "M", "F", or "I" (infant)
    pub length: Vec<f64>,           // longest shell measurement (mm)
    pub diameter: Vec<f64>,         // perpendicular to length (mm)
    pub height: Vec<f64>,           // with meat in shell (mm)
    pub whole_weight: Vec<f64>,     // whole abalone (g)
    pub shucked_weight: Vec<f64>,   // weight of meat (g)
    pub viscera_weight: Vec<f64>,   // gut weight after bleeding (g)
    pub shell_weight: Vec<f64>,     // after drying (g)
    pub rings: Vec<u8>,             // target variable — rings + 1.5 = age in years
}



impl AbaloneDataset {
    // Parses a raw CSV string (no headers) into an AbaloneDataset.
    // Invalid rows are silently skipped: wrong field count, unknown sex value,
    // or any field that fails to parse as a number.
    pub fn parse(csv: &str) -> Self {
        // Start with empty Vecs — we'll grow them one valid row at a time
        let mut dataset = Self {
            sex: Vec::new(),
            length: Vec::new(),
            diameter: Vec::new(),
            height: Vec::new(),
            whole_weight: Vec::new(),
            shucked_weight: Vec::new(),
            viscera_weight: Vec::new(),
            shell_weight: Vec::new(),
            rings: Vec::new(),
        };

        for line in csv.lines() {
            // Split the line on commas and collect into a temporary Vec of string slices.
            // &str is a borrowed string slice — no allocation, just a view into the line.
            let fields: Vec<&str> = line.split(',').collect();

            // The dataset has exactly 9 columns — skip anything malformed
            if fields.len() != 9 {
                continue;
            }

            // Validate sex before trying to parse numbers
            let sex = fields[0].trim().to_string();
            if !["M", "F", "I"].contains(&sex.as_str()) {
                continue;
            }

            // Parse all 8 numerical fields at once into a tuple of Results.
            // Each .parse::<f64>() returns Ok(value) on success or Err on failure.
            let parsed = (
                fields[1].trim().parse::<f64>(),
                fields[2].trim().parse::<f64>(),
                fields[3].trim().parse::<f64>(),
                fields[4].trim().parse::<f64>(),
                fields[5].trim().parse::<f64>(),
                fields[6].trim().parse::<f64>(),
                fields[7].trim().parse::<f64>(),
                fields[8].trim().parse::<u8>(),
            );

            // if let destructures the tuple — this only executes if ALL fields
            // parsed successfully. If any single field is Err, the whole row is skipped.
            if let (Ok(length), Ok(diameter), Ok(height),
                Ok(ww), Ok(sw), Ok(vw), Ok(shw), Ok(rings)) = parsed {
                dataset.sex.push(sex);
                dataset.length.push(length);
                dataset.diameter.push(diameter);
                dataset.height.push(height);
                dataset.whole_weight.push(ww);
                dataset.shucked_weight.push(sw);
                dataset.viscera_weight.push(vw);
                dataset.shell_weight.push(shw);
                dataset.rings.push(rings);
            }
        }

        dataset
    }

    pub fn numerical_columns(&self) -> Vec<(&str, Vec<f64>)> {
        vec![
            ("length",         self.length.clone()),
            ("diameter",       self.diameter.clone()),
            ("height",         self.height.clone()),
            ("whole_weight",   self.whole_weight.clone()),
            ("shucked_weight", self.shucked_weight.clone()),
            ("viscera_weight", self.viscera_weight.clone()),
            ("shell_weight",   self.shell_weight.clone()),
            ("rings",          self.rings.iter().map(|&r| r as f64).collect()),
        ]
    }

    // Number of valid samples in the dataset
    pub fn len(&self) -> usize {
        self.rings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rings.is_empty()
    }
}