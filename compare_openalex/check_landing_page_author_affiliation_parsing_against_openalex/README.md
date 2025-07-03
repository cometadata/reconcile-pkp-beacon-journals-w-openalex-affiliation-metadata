# Check Landing Page Authors and Affiliations against OpenAlex

Validates author and institution data from landing page parsing against OpenAlex API records.

## Usage

```bash
python check_landing_page_author_affiliation_parsing_against_openalex.py -i input.csv [-o output.csv]
```

## Required Input CSV Columns
- `RelativeFilePath`
- `DOI`
- `Author`
- `Institution`

## Output
Adds comparison columns:
- `exact_author_match` - Exact string match
- `normalized_author_match` - Case/accent-insensitive match
- `author_similarity_score` - Fuzzy match score (0-100)
- `exact_institution_match` - Exact string match
- `normalized_institution_match` - Case/accent-insensitive match
- `institution_similarity_score` - Fuzzy match score (0-100)
- `openalex_matched_author_name` - Best matching author from OpenAlex
- `openalex_matched_institution_strings` - Affiliated institutions from OpenAlex
