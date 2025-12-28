const fs = require('fs');
const path = require('path');

const inputFileName = 'mcq_extraction_results (2).json';
const outputFileName = 'mcq_extraction_results (2).json';

const filePath = path.join(__dirname, inputFileName);

try {
    if (!fs.existsSync(filePath)) {
        throw new Error(`File not found: ${filePath}`);
    }

    const rawData = fs.readFileSync(filePath, 'utf8');
    let mcqData = JSON.parse(rawData);

    function updateValues(data, key, value) {
        if (!Array.isArray(data)) {
            console.error("Provided data is not an array.");
            return data;
        }

        console.log(`Processing ${data.length} items...`);
        
        data.forEach(item => {
            item[key] = value;
        });

        console.log(`Updated key '${key}' with value '${value}' for all items.`);
        return data;
    }
    
    const args = process.argv.slice(2);
    const targetKey = args[0] || "chapter";
    const targetValue = args[1] || "Principle of Mathematical Induction";

    const updatedData = updateValues(mcqData, targetKey, targetValue);

    const outputFilePath = path.join(__dirname, outputFileName);
    fs.writeFileSync(outputFilePath, JSON.stringify(updatedData, null, 2), 'utf8');

    console.log(`✅ Success! Data saved to ${outputFileName}`);

} catch (error) {
    console.error("Error processing file:", error.message);
}
