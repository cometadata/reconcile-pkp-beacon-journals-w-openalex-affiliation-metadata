# PDF URL Validator and Downloader

Script for validating URLs, detecting PDFs, and downloading files from a CSV dataset.

## Overview

This script processes a CSV file containing URLs, checks if they resolve to PDF documents, downloads the PDFs when possible, and logs the results in a new CSV file.

## Requirements

- Python 3.x
- Dependencies: requests

## Usage

```bash
python pdf_url_validator.py -i <input_csv> [-o <output_csv>] [-d <download_directory>]
```

### Parameters

- `-i, --input`: Path to input CSV file
- `-o, --output`: Path for output CSV file (default: [input_filename]_processed.csv)
- `-d, --downloaddir`: Directory to save downloaded PDFs (default: [input_filename]_pdfs/)


## Notes

- Allows unverified HTTPS requests, suppressing SSL warnings
- Creates download directory if it doesn't exist
- Outputs processing status to console