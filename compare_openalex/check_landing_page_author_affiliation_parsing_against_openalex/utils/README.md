# Get Stats

Calculates statistics from check_landing_page_author_affiliation_parsing_against_openalex.py output.

## Usage

```bash
python get_stats.py -i comparison_results.csv [-o1 overall_stats.csv] [-o2 per_doi_stats.csv]
```

## Input
Expects CSV with columns:
- `DOI`
- `exact_author_match`
- `normalized_author_match`
- `author_similarity_score`
- `exact_institution_match`
- `normalized_institution_match`
- `institution_similarity_score`

## Outputs
The script putputs two CSVs:
- Overall statistics: Contains match counts and percentages, score statistics (mean, median, min, max, stdev), and highlights a small set of special cases (e.g., high scores without matches)
- Per-DOI statistics: Author/institution match counts per DOI
