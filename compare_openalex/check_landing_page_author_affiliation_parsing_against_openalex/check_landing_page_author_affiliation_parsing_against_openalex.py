import re
import csv
import sys
import json
import time
import string
import requests
import argparse
from datetime import datetime
from rapidfuzz import fuzz
from unidecode import unidecode

OPENALEX_API_BASE_URL = "https://api.openalex.org/works/https://doi.org/"


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    try:
        text = unidecode(text)
    except Exception as e:
        print(f"Warning: unidecode failed for text '{text[:50]}...': {e}", file=sys.stderr)
        pass
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = ' '.join(text.split())
    return text


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Verify author/institution matches using OpenAlex API with multiple comparison methods (Exact, Normalized, Edit Distance) and write results iteratively, logging matched OpenAlex data."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to the input CSV file."
                        )
    parser.add_argument("-o", "--output", required=False, default=None, help="Path for the output CSV file. If omitted, a default name with timestamp is generated."
                        )
    return parser.parse_args()


def generate_default_output_filename():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"comparison_results_{timestamp}.csv"


def read_input_csv(filepath):
    grouped_data = {}
    required_columns = ['RelativeFilePath', 'DOI', 'Author', 'Institution']
    fieldnames = []
    try:
        with open(filepath, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            if not reader.fieldnames:
                print(f"Error: Input CSV file '{filepath}' is empty or has no header.", file=sys.stderr)
                return None, None
            fieldnames = reader.fieldnames

            if not all(col in fieldnames for col in required_columns):
                missing = [
                    col for col in required_columns if col not in fieldnames]
                print(f"Error: Input CSV missing required columns: {', '.join(missing)}", file=sys.stderr)
                return None, None

            for i, row in enumerate(reader):
                doi = row.get('DOI')
                if doi:
                    if not isinstance(doi, str) or not doi.strip().startswith('10.'):
                        print(f"Warning: Skipping row {i+2} due to invalid DOI format: '{doi}'", file=sys.stderr)
                        continue

                    doi = doi.strip()
                    if doi not in grouped_data:
                        grouped_data[doi] = []
                    grouped_data[doi].append(row)
                else:
                    print(f"Warning: Skipping row {i+2} due to missing DOI: {row}", file=sys.stderr)

            if not grouped_data:
                print(f"Warning: No valid rows with DOIs found in '{filepath}'.", file=sys.stderr)
                return None, fieldnames if fieldnames else None

            return grouped_data, fieldnames

    except FileNotFoundError:
        print(f"Error: Input file not found at '{filepath}'", file=sys.stderr)
        return None, None
    except Exception as e:
        print(f"Error reading CSV file '{filepath}': {e}", file=sys.stderr)
        return None, None


def fetch_openalex_data(doi):
    if doi.startswith("https://doi.org/"):
        doi_cleaned = doi.replace("https://doi.org/", "", 1)
    elif doi.startswith("doi:"):
        doi_cleaned = doi.replace("doi:", "", 1)
    else:
        doi_cleaned = doi

    url = f"{OPENALEX_API_BASE_URL}{doi_cleaned}"
    print(f"Fetching: {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        print(f"Error: Request timed out for DOI {doi}", file=sys.stderr)
        return None
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"Info: DOI {doi} not found in OpenAlex (404). URL: {url}", file=sys.stderr)
        else:
            print(f"Error: HTTP error fetching DOI {doi}: {e}. URL: {url}", file=sys.stderr)
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON response for DOI {doi}. URL: {url}. Response text: {response.text[:200]}...", file=sys.stderr)
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error: Network or request error fetching DOI {doi}: {e}. URL: {url}", file=sys.stderr)
        return None


def process_and_write_data(grouped_data, writer):
    total_dois = len(grouped_data)
    processed_dois = 0
    rows_written = 0

    for doi, input_rows in grouped_data.items():
        processed_dois += 1
        print(f"Processing DOI {processed_dois}/{total_dois}: {doi}")

        openalex_json = fetch_openalex_data(doi)
        time.sleep(0.15)

        authorships_api = []
        if openalex_json and isinstance(openalex_json.get('authorships'), list):
            authorships_api = openalex_json.get('authorships', [])
        elif openalex_json:
            print(f"Warning: 'authorships' key missing or not a list in OpenAlex response for DOI {doi}. JSON keys: {list(openalex_json.keys())}", file=sys.stderr)

        for input_row in input_rows:
            author_name_csv_raw = input_row.get('Author', '')
            institution_csv_raw = input_row.get('Institution', '')

            author_name_csv_normalized = normalize_text(
                author_name_csv_raw)
            institution_csv_normalized = normalize_text(
                institution_csv_raw)

            best_author_score = -1.0
            best_match_idx = -1

            if author_name_csv_normalized and authorships_api:
                for idx, authorship in enumerate(authorships_api):
                    author_name_api_raw = authorship.get('raw_author_name', '')
                    if not author_name_api_raw:
                        author_name_api_raw = authorship.get(
                            'author', {}).get('display_name', '')

                    author_name_api_normalized = normalize_text(
                        author_name_api_raw)

                    current_author_score = fuzz.ratio(
                        author_name_csv_normalized, author_name_api_normalized)

                    if current_author_score > best_author_score:
                        best_author_score = current_author_score
                        best_match_idx = idx
                        if best_author_score == 100.0:
                            break

            exact_author_match = False
            normalized_author_match = False
            author_similarity_score = 0.0
            exact_institution_match = False
            normalized_institution_match = False
            institution_similarity_score = 0.0
            openalex_author_name_matched = ""
            openalex_institutions_matched_str = ""

            if best_match_idx != -1:
                best_authorship = authorships_api[best_match_idx]

                best_author_api_raw_name = best_authorship.get(
                    'raw_author_name')
                if not best_author_api_raw_name:
                    best_author_api_raw_name = best_authorship.get(
                        'author', {}).get('display_name', '')

                openalex_author_name_matched = best_author_api_raw_name if best_author_api_raw_name else ""

                best_author_api_normalized = normalize_text(
                    best_author_api_raw_name)

                author_similarity_score = best_author_score

                exact_author_match = (
                    author_name_csv_raw == best_author_api_raw_name)

                normalized_author_match = (
                    author_name_csv_normalized == best_author_api_normalized)

                raw_affiliations_api = best_authorship.get(
                    'raw_affiliation_strings', []) or []

                openalex_institutions_matched_str = '; '.join(
                    raw_affiliations_api)

                normalized_affiliations_api = [
                    normalize_text(aff) for aff in raw_affiliations_api if aff
                ]
                normalized_affiliations_api = [
                    aff for aff in normalized_affiliations_api if aff]

                if not institution_csv_raw:
                    exact_institution_match = not bool(raw_affiliations_api)
                elif raw_affiliations_api:
                    exact_institution_match = any(institution_csv_raw.lower(
                    ) == api_aff.lower() for api_aff in raw_affiliations_api)

                if not institution_csv_normalized:
                    normalized_institution_match = not bool(
                        normalized_affiliations_api)
                elif normalized_affiliations_api:
                    normalized_institution_match = institution_csv_normalized in normalized_affiliations_api

                if not institution_csv_normalized:
                    institution_similarity_score = 100.0 if not normalized_affiliations_api else 0.0
                elif normalized_affiliations_api:
                    max_inst_score = 0.0
                    for norm_api_aff in normalized_affiliations_api:
                        score = fuzz.ratio(
                            institution_csv_normalized, norm_api_aff)
                        max_inst_score = max(max_inst_score, score)
                    institution_similarity_score = max_inst_score
                else:
                    institution_similarity_score = 0.0

            else:
                if not author_name_csv_normalized:
                    author_similarity_score = 0.0
                else:
                    author_similarity_score = 0.0

                if not institution_csv_normalized:
                    institution_similarity_score = 0.0
                else:
                    institution_similarity_score = 0.0

            output_row = {}
            output_row.update(input_row)
            output_row['exact_author_match'] = exact_author_match
            output_row['normalized_author_match'] = normalized_author_match
            output_row['author_similarity_score'] = round(
                author_similarity_score, 2)
            output_row['exact_institution_match'] = exact_institution_match
            output_row['normalized_institution_match'] = normalized_institution_match
            output_row['institution_similarity_score'] = round(
                institution_similarity_score, 2)
            output_row['openalex_matched_author_name'] = openalex_author_name_matched
            output_row['openalex_matched_institution_strings'] = openalex_institutions_matched_str

            try:
                writer.writerow(output_row)
                rows_written += 1
            except Exception as e:
                print(f"Error writing row to CSV for DOI {doi}, Author '{author_name_csv_raw}'. \nRow Data: {output_row}\nError: {e}", file=sys.stderr)

    return rows_written


def main():
    args = parse_arguments()

    print(f"Reading input CSV: {args.input}")
    grouped_data, input_fieldnames = read_input_csv(args.input)

    if grouped_data is None or input_fieldnames is None:
        print("Exiting due to input file errors or empty input.")
        sys.exit(1)

    output_filename = args.output if args.output else generate_default_output_filename()
    print(f"Output will be written to: {output_filename}")

    added_fieldnames = [
        'exact_author_match', 'normalized_author_match', 'author_similarity_score',
        'exact_institution_match', 'normalized_institution_match', 'institution_similarity_score',
        'openalex_matched_author_name', 'openalex_matched_institution_strings'
    ]

    output_fieldnames = input_fieldnames + \
        [f for f in added_fieldnames if f not in input_fieldnames]

    total_rows_written = 0
    try:
        with open(output_filename, mode='w', encoding='utf-8') as outfile:
            writer = csv.DictWriter(
                outfile, fieldnames=output_fieldnames, extrasaction='ignore')
            writer.writeheader()

            print("Starting data processing and OpenAlex API queries...")
            total_rows_written = process_and_write_data(grouped_data, writer)

        print(f"\nSuccessfully wrote {total_rows_written} rows to '{output_filename}'")

    except IOError as e:
        print(f"Error opening or writing output file '{output_filename}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred during processing or writing: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)

    print("Script finished.")


if __name__ == "__main__":
    main()
