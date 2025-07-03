# PDF URL Validator and Downloader

Script for validating URLs, detecting PDFs, and downloading files from a CSV dataset containing PKP journal article PDF URLs derived from OpenAlex.

## Usage

```bash
python retrieve_pkp_work_url_download_pdf.py -i <input_csv> [-o <output_csv>] [-d <download_directory>]
```

### Parameters

- `-i, --input`: Path to input CSV file
- `-o, --output`: Path for output CSV file (default: [input_filename]_processed.csv)
- `-d, --directory`: Directory to save downloaded PDFs (default: [input_filename]_pdfs/)