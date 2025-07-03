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

### Overall Statistics (-o1)
- Match counts and percentages
- Score statistics (mean, median, min, max, stdev)
- Special cases (e.g., high scores without matches)

### Per-DOI Statistics (-o2)
- Author/institution match counts per DOI
- Score statistics per DOI
- Flags for all/any matches
