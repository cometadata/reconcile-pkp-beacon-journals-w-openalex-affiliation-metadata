### PKP Beacon Journal and OpenAlex Data Comparison Pipeline

We begin with the `get_journal_base_urls_from_pkp_beacon_file.py` script, which takes the [PKP Beacon data file](https://doi.org/10.7910/DVN/OCZNVY) as input. From this, we processes the `oai_url` column to produce a deduplicated CSV file containing a single `base_url` column for each journal.

This list of base URLs is then used as input for the `get-openalex-works-with-pkp-beacon-base-url` Rust utility, alongside [a full OpenAlex data snapshot](https://docs.openalex.org/download-all-data/openalex-snapshot). We filter the snapshot to the set of gzipped JSONL files containing only the works PKP Beacon journals. Then, using the  `parse_record_data_from_files.py` script, we process these filtered files, converting them into a single master CSV that includes the basic metadata needed to retrieve the landing pages for each work.

To create a more manageable set of landing pages for comparison/analysis, we sample a random subset of PKP Beacon journal articles from the CSV using the `sample.py` script.

Using this random subset of PKP Beacon journal articles, we use `retrieve_landing_pages.py` to saves the HTML content from the URL of each article's landing page. These HTML files are then processed by `parse_landing_page_html.py` to  extract the DOI, author, and affiliation data data (if present on the landing apge), creating a new CSV file containing this information.

The core comparison is then performed using `check_landing_page_author_affiliation_parsing_against_openalex.py` . Here, we take the CSV of parsed landing page data and, for each DOI, query the OpenAlex API to retrieve the corresponding work record. We then compares the author and institution strings from the landing page against the OpenAlex data using a few different methods, namely:
- A direct, case-sensitive comparison of the authors and affiliations.
- A comparison of the authors and affiliations after text is lowercased, whitespace is normalized, and its converted to ASCII format.
- Similarity scores using the `rapidfuzz` library.

With `check_landing_page_author_affiliation_parsing_against_openalex.py` outputting an enriched form of the parsed landing page data CSV that includes the new columns for each comparison result.

Finally, we use `get_stats.py` to generates aggregate statistics from the comparison results CSV. This calculates several metrics, outputting them into two summary files:
- An overall statistics CSV, which includes total match counts, percentages, and statistical analysis of the similarity scores (mean, median, standard deviation).
- A per-DOI statistics CSV, which provides a granular breakdown of author and institution match counts for each individual work.
