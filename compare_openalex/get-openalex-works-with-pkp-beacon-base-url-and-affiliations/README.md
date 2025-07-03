# Get OpenAlex works with PKP Beacon base URL

Rust utility to filter OpenAlex snapshot files to those records containing PKP Beacon journal URLs and affiliations


## Options

- `--input-dir`: Directory containing OpenAlex JSONL.gz files (recursive search)
- `--output-dir`: Base directory for organized output
- `--base-urls-csv`: CSV file with base URLs to match against (header: "base_url")
- `--max-open-prefix-files`: Maximum number of prefix output files to keep open (default: 256)
- `--log-level`: Logging detail level: DEBUG, INFO, WARN, ERROR (default: INFO)
- `--threads`: Number of processing threads (0 = auto-detect cores, default: 0)
- `--stats-interval`: Seconds between statistics logging (default: 60)


## Usage

```bash
get-openalex-works-with-pkp-beacon-base-url \
  --input-dir /path/to/openalex/files \
  --output-dir /path/to/output \
  --base-urls-csv urls.csv \
  [--max-open-prefix-files 256] \
  [--log-level INFO] \
  [--threads 0] \
  [--stats-interval 60]
```

## Filtering Criteria

1. Matches any `locations[].landing_page_url` against base URLs in the PKP Beacon CSV input file
2. At least one non-empty `authorships[].raw_affiliation_strings` entry


## Input File Format

The base URLs CSV must have a header row with "base_url" and contain fully qualified URLs:

```csv
base_url
https://example.com
https://journal.university.edu
```

The innput file can be derived from  get_journal_base_urls_from_pkp_beacon_file.py.

## Output Structure

Files are organized in sub-directories by prefix in the output directory, e.g.:

```
output_dir/
├── 10.1000/
│   └── data.jsonl.gz
├── 10.1234/
│   └── data.jsonl.gz
└── _unknown_/
    └── data.jsonl.gz
```

