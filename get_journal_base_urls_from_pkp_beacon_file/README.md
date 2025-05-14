# Get Journal Base URLs from PKP Beacon file

Extracts and deduplicates base journal URLs from the PKP Beacon data file.

## Usage

```bash
python get_journal_base_urls_from_pkp_beacon_file.py -i <input_file.csv> -o <output_file.csv>
````

**Arguments:**

  * `-i`, `--input`: Path to the input CSV file. Must contain an `oai_url` column.
  * `-o`, `--output`: Path for the output CSV file.

## Input

[PKP Beacon data file](https://doi.org/10.7910/DVN/OCZNVY) with an `oai_url` column. Example:

```csv
oai_url,other_columns
[https://example.com/journal/index.php/index/oai](https://example.com/journal/index.php/index/oai),...
[https://another.org/ojs/index.php/index/oai](https://another.org/ojs/index.php/index/oai),...
[https://some.site/oai](https://some.site/oai),...
```

## Output

A CSV file with a single `base_url` column, listing theunique URLs. Example:

```csv
base_url
[https://example.com](https://example.com)
[https://another.org/ojs](https://another.org/ojs)
[https://some.site](https://some.site)
```
