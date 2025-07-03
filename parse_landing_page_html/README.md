# Parse Landing Page HTML

Extracts author, institution, and DOI metadata from the HTML of PKP Beacon journal landing pages.

## Installation

```bash
pip install beautifulsoup4
```

## Usage

```bash
python parse_landing_page_html.py -i <input_dir> [-l log.csv] [-o output.csv] [-s stats.csv]
```

## Output Files

The script outputs three CSVs:

- Log file (`processing_log_TIMESTAMP.csv`): Tracks the parsing of each HTML file, including whether citation meta tags were found and any errors encountered
- Output file(`author_affiliation_doi_merged_TIMESTAMP.csv`): Contains extracted author-institution-DOI data with columns: RelativeFilePath, DOI, Author, Institution, Source (meta/ul.authors)
- Stats file (`parsing_stats_TIMESTAMP.csv`): Summary statistics for total files processed, tag presence percentages, unique authors/institutions counts, and breakdowns of the tag derivations for authors/institutions.

