# retrieve_landing_pages.py

Fetches and saves HTML files from PKP Beacon journal landing pages.

## Usage
```bash
python retrieve_landing_pages.py -i input.csv -d output_dir [-l log.log] [-w 10]
```

## Arguments
- `-i, --input`: Input CSV file path (required)
- `-d, --data-dir`: Output directory for HTML files (required)
- `-l, --log-file`: Log file path (default: log.log)
- `-w, --workers`: Concurrent workers (default: 10)

## Input CSV Format
Required columns:
- `url`: Target URL
- `member_id`: Member identifier
- `prefix`: DOI prefix
- `doi`: Digital Object Identifier
- `line_number`: Row number

## Output
- HTML files saved in: `{data-dir}/{member_id}/{sanitized_prefix}/{base64_doi}.html`
- Log CSV with fetch status and file paths