# Data Parser for Crossref and OpenAlex Records

## Overview
Python script to extract metadata from Crossref or OpenAlex gzipped JSONL files/


## Installation
```bash
python pip install tqdm
```

## Usage
```bash
python parse_record_data_from_files.py -i /path/to/data/directory -s [crossref|openalex] -o output.csv
```

### Arguments
- `-i, --input-dir`: Base directory containing gzipped JSONL files (required)
- `-s, --source-type`: Source format, either "crossref" or "openalex" (required)
- `-o, --output-csv`: Path for output CSV file (default: extracted_data.csv)
- `-l, --log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `-m, --max-records-per-file`: Process only N records per file (for testing)

## Output
The script generates a CSV with these columns:
- member_id
- prefix
- doi
- url
- pdf_url (if source is OpenAlex)
- deposit_timestamp_yyyymmddhhmmss
- openalex_id
- source_file_relative
- line_number
- parse_status
- parse_error_details