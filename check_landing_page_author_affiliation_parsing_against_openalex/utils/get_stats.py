import csv
import sys
import argparse
import statistics
from datetime import datetime
from collections import defaultdict


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Calculate overall and per-DOI statistics on matching results.")
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the input CSV file (output from the comparison script).")
    parser.add_argument(
        "-o1", "--output-overall", required=False, default=None, help="Path for the overall statistics CSV file. Default: comparison_stats_overall_TIMESTAMP.csv")
    parser.add_argument(
        "-o2", "--output-per-doi", required=False, default=None, help="Path for the per-DOI statistics CSV file. Default: comparison_stats_per_doi_TIMESTAMP.csv")
    return parser.parse_args()


def generate_default_filenames():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    overall_filename = f"comparison_stats_overall_{timestamp}.csv"
    per_doi_filename = f"comparison_stats_per_doi_{timestamp}.csv"
    return overall_filename, per_doi_filename


def safe_float_convert(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_bool_convert(value):
    return str(value).strip().lower() == 'true'


def calculate_overall_statistics(filepath):
    required_cols = [
        'DOI', 'exact_author_match', 'normalized_author_match', 'author_similarity_score',
        'exact_institution_match', 'normalized_institution_match', 'institution_similarity_score'
    ]

    total_rows = 0
    counts = defaultdict(int)
    author_scores = []
    institution_scores = []
    skipped_rows = 0

    try:
        with open(filepath, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)

            if not reader.fieldnames or not all(col in reader.fieldnames for col in required_cols):
                missing = [col for col in required_cols if col not in (
                    reader.fieldnames or [])]
                print(f"Error: Input CSV header missing required columns for overall stats: {', '.join(missing)}", file=sys.stderr)
                return None

            for i, row in enumerate(reader):
                total_rows += 1
                try:
                    exact_author = safe_bool_convert(
                        row.get('exact_author_match'))
                    norm_author = safe_bool_convert(
                        row.get('normalized_author_match'))
                    exact_inst = safe_bool_convert(
                        row.get('exact_institution_match'))
                    norm_inst = safe_bool_convert(
                        row.get('normalized_institution_match'))

                    author_score = safe_float_convert(
                        row.get('author_similarity_score'))
                    inst_score = safe_float_convert(
                        row.get('institution_similarity_score'))

                    if exact_author:
                        counts['exact_author_match'] += 1
                    if norm_author:
                        counts['normalized_author_match'] += 1
                    if exact_inst:
                        counts['exact_institution_match'] += 1
                    if norm_inst:
                        counts['normalized_institution_match'] += 1
                    if exact_author and exact_inst:
                        counts['exact_both_match'] += 1
                    if norm_author and norm_inst:
                        counts['normalized_both_match'] += 1

                    author_scores.append(author_score)
                    institution_scores.append(inst_score)

                    if norm_author and not exact_author:
                        counts['norm_author_not_exact'] += 1
                    if norm_inst and not exact_inst:
                        counts['norm_inst_not_exact'] += 1
                    if author_score > 90.0 and not norm_author:
                        counts['high_author_score_no_norm_match'] += 1
                    if inst_score > 90.0 and not norm_inst:
                        counts['high_inst_score_no_norm_match'] += 1
                    if norm_author and not norm_inst:
                        counts['norm_author_match_no_norm_inst'] += 1
                    if author_score == 100.0:
                        counts['perfect_author_score'] += 1
                    if inst_score == 100.0:
                        counts['perfect_inst_score'] += 1

                except Exception as e:
                    print(f"Warning: Error processing row {i+1} ({e}). Skipping row for overall stats.", file=sys.stderr)
                    skipped_rows += 1
                    continue

        valid_rows = total_rows - skipped_rows
        if valid_rows == 0:
            print("No valid data rows found for overall statistics.", file=sys.stderr)
            return None

        stats_results = {'total_rows': total_rows,
                         'valid_rows': valid_rows, 'skipped_rows': skipped_rows}

        for key, count in counts.items():
            stats_results[f'{key}_count'] = count
            stats_results[f'{key}_pct'] = (count / valid_rows) * 100 if valid_rows > 0 else 0.0

        def calculate_score_stats(scores, prefix):
            results = {f'{prefix}_mean': None, f'{prefix}_median': None, f'{prefix}_min': None, f'{prefix}_max': None, f'{prefix}_stdev': None}
            valid_scores = [s for s in scores if isinstance(s, (int, float))]
            if valid_scores:
                results[f'{prefix}_mean'] = statistics.mean(valid_scores)
                results[f'{prefix}_median'] = statistics.median(valid_scores)
                results[f'{prefix}_min'] = min(valid_scores)
                results[f'{prefix}_max'] = max(valid_scores)
                results[f'{prefix}_stdev'] = statistics.stdev(valid_scores) if len(valid_scores) > 1 else 0.0
            return results

        stats_results.update(calculate_score_stats(
            author_scores, 'author_score'))
        stats_results.update(calculate_score_stats(
            institution_scores, 'inst_score'))

        return stats_results

    except FileNotFoundError:
        print(f"Error: Input file not found at '{filepath}'", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error reading or processing CSV file '{filepath}': {e}", file=sys.stderr)
        return None


def calculate_per_doi_statistics(filepath):
    required_cols = [
        'DOI', 'exact_author_match', 'normalized_author_match', 'author_similarity_score',
        'exact_institution_match', 'normalized_institution_match', 'institution_similarity_score'
    ]
    doi_data = defaultdict(lambda: {'rows': [], 'skipped': 0})

    try:
        with open(filepath, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)

            if not reader.fieldnames or not all(col in reader.fieldnames for col in required_cols):
                missing = [col for col in required_cols if col not in (
                    reader.fieldnames or [])]
                print(f"Error: Input CSV header missing required columns for per-DOI stats: {', '.join(missing)}", file=sys.stderr)
                return None

            for i, row in enumerate(reader):
                doi = row.get('DOI', '').strip()
                if not doi:
                    print(f"Warning: Skipping row {i+1} due to missing DOI for per-DOI stats.", file=sys.stderr)
                    if 'UNKNOWN_DOI' not in doi_data:
                        doi_data['UNKNOWN_DOI'] = {'rows': [], 'skipped': 0}
                    doi_data['UNKNOWN_DOI']['skipped'] += 1
                    continue

                try:
                    row_data = {
                        'exact_author': safe_bool_convert(row.get('exact_author_match')),
                        'norm_author': safe_bool_convert(row.get('normalized_author_match')),
                        'author_score': safe_float_convert(row.get('author_similarity_score')),
                        'exact_inst': safe_bool_convert(row.get('exact_institution_match')),
                        'norm_inst': safe_bool_convert(row.get('normalized_institution_match')),
                        'inst_score': safe_float_convert(row.get('institution_similarity_score'))
                    }
                    doi_data[doi]['rows'].append(row_data)
                except Exception as e:
                    print(f"Warning: Error processing row {i+1} for DOI {doi} ({e}). Skipping row.", file=sys.stderr)
                    doi_data[doi]['skipped'] += 1
                    continue

        if not doi_data:
            print("No valid DOI data found to process.", file=sys.stderr)
            return None

        per_doi_results = []
        for doi, data in doi_data.items():
            rows = data['rows']
            skipped = data['skipped']
            total_authors = len(rows)

            if total_authors == 0 and skipped == 0:
                continue

            doi_stats = {
                'DOI': doi, 'TotalAuthorsInput': total_authors, 'SkippedRows': skipped}

            if total_authors > 0:
                doi_stats['CountExactAuthorMatch'] = sum(
                    r['exact_author'] for r in rows)
                doi_stats['CountNormAuthorMatch'] = sum(
                    r['norm_author'] for r in rows)
                doi_stats['CountExactInstMatch'] = sum(
                    r['exact_inst'] for r in rows)
                doi_stats['CountNormInstMatch'] = sum(
                    r['norm_inst'] for r in rows)
                doi_stats['CountExactBothMatch'] = sum(
                    r['exact_author'] and r['exact_inst'] for r in rows)
                doi_stats['CountNormBothMatch'] = sum(
                    r['norm_author'] and r['norm_inst'] for r in rows)

                author_scores = [r['author_score'] for r in rows]
                inst_scores = [r['inst_score'] for r in rows]

                doi_stats['AvgAuthorScore'] = statistics.mean(
                    author_scores) if author_scores else 0.0
                doi_stats['MedianAuthorScore'] = statistics.median(
                    author_scores) if author_scores else 0.0
                doi_stats['MinAuthorScore'] = min(
                    author_scores) if author_scores else 0.0
                doi_stats['MaxAuthorScore'] = max(
                    author_scores) if author_scores else 0.0

                doi_stats['AvgInstScore'] = statistics.mean(
                    inst_scores) if inst_scores else 0.0
                doi_stats['MedianInstScore'] = statistics.median(
                    inst_scores) if inst_scores else 0.0
                doi_stats['MinInstScore'] = min(
                    inst_scores) if inst_scores else 0.0
                doi_stats['MaxInstScore'] = max(
                    inst_scores) if inst_scores else 0.0

                doi_stats['AllAuthorsExactMatch'] = doi_stats['CountExactAuthorMatch'] == total_authors
                doi_stats['AllAuthorsNormMatch'] = doi_stats['CountNormAuthorMatch'] == total_authors
                doi_stats['AnyAuthorExactMatch'] = doi_stats['CountExactAuthorMatch'] > 0
                doi_stats['AnyAuthorNormMatch'] = doi_stats['CountNormAuthorMatch'] > 0
                doi_stats['AllInstExactMatch'] = doi_stats['CountExactInstMatch'] == total_authors
                doi_stats['AllInstNormMatch'] = doi_stats['CountNormInstMatch'] == total_authors
            else:
                keys_to_zero = ['CountExactAuthorMatch', 'CountNormAuthorMatch', 'CountExactInstMatch',
                                'CountNormInstMatch', 'CountExactBothMatch', 'CountNormBothMatch',
                                'AvgAuthorScore', 'MedianAuthorScore', 'MinAuthorScore', 'MaxAuthorScore',
                                'AvgInstScore', 'MedianInstScore', 'MinInstScore', 'MaxInstScore']
                keys_to_false = ['AllAuthorsExactMatch', 'AllAuthorsNormMatch', 'AnyAuthorExactMatch',
                                 'AnyAuthorNormMatch', 'AllInstExactMatch', 'AllInstNormMatch']
                for k in keys_to_zero:
                    doi_stats[k] = 0.0
                for k in keys_to_false:
                    doi_stats[k] = False

            per_doi_results.append(doi_stats)

        return per_doi_results

    except FileNotFoundError:
        print(f"Error: Input file not found at '{filepath}'", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error reading or processing CSV file '{filepath}': {e}", file=sys.stderr)
        return None


def write_overall_stats_csv(filepath, stats):
    if stats is None:
        print("Overall statistics calculation failed. No output file written.", file=sys.stderr)
        return False

    try:
        with open(filepath, mode='w', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['Statistic_Name', 'Value'])

            for key, value in stats.items():
                formatted_value = ""
                if value is None:
                    formatted_value = "N/A"
                elif isinstance(value, float):
                    if key.endswith('_pct') or 'score' in key:
                        formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                else:
                    formatted_value = str(value)

                writer.writerow([key, formatted_value])
        return True
    except IOError as e:
        print(f"Error writing overall statistics CSV file '{filepath}': {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred during overall CSV writing: {e}", file=sys.stderr)
        return False


def write_per_doi_stats_csv(filepath, doi_stats_list):
    if doi_stats_list is None or not doi_stats_list:
        print("Per-DOI statistics calculation failed or yielded no results. No output file written.", file=sys.stderr)
        return False

    fieldnames = [
        'DOI', 'TotalAuthorsInput', 'SkippedRows',
        'CountExactAuthorMatch', 'CountNormAuthorMatch',
        'CountExactInstMatch', 'CountNormInstMatch',
        'CountExactBothMatch', 'CountNormBothMatch',
        'AvgAuthorScore', 'MedianAuthorScore', 'MinAuthorScore', 'MaxAuthorScore',
        'AvgInstScore', 'MedianInstScore', 'MinInstScore', 'MaxInstScore',
        'AllAuthorsExactMatch', 'AllAuthorsNormMatch',
        'AnyAuthorExactMatch', 'AnyAuthorNormMatch',
        'AllInstExactMatch', 'AllInstNormMatch'
    ]

    try:
        with open(filepath, mode='w', encoding='utf-8') as outfile:
            writer = csv.DictWriter(
                outfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for row_dict in doi_stats_list:
                for key in row_dict:
                    if isinstance(row_dict[key], float):
                        row_dict[key] = f"{row_dict[key]:.2f}"
                writer.writerow(row_dict)
        return True
    except IOError as e:
        print(f"Error writing per-DOI statistics CSV file '{filepath}': {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred during per-DOI CSV writing: {e}", file=sys.stderr)
        return False


def main():
    args = parse_arguments()
    default_overall_filename, default_per_doi_filename = generate_default_filenames()

    overall_output_filename = args.output_overall if args.output_overall else default_overall_filename
    per_doi_output_filename = args.output_per_doi if args.output_per_doi else default_per_doi_filename

    print(f"Input file: {args.input}")
    print(f"Output file (Overall): {overall_output_filename}")
    print(f"Output file (Per-DOI): {per_doi_output_filename}")

    print("\nCalculating overall statistics...")
    overall_stats_data = calculate_overall_statistics(args.input)
    if overall_stats_data:
        success_overall = write_overall_stats_csv(
            overall_output_filename, overall_stats_data)
        if success_overall:
            print(f"Successfully wrote overall statistics to '{overall_output_filename}'")
    else:
        print("No overall statistics generated.")

    print("\nCalculating per-DOI statistics...")
    per_doi_stats_data = calculate_per_doi_statistics(args.input)
    if per_doi_stats_data:
        success_per_doi = write_per_doi_stats_csv(
            per_doi_output_filename, per_doi_stats_data)
        if success_per_doi:
            print(f"Successfully wrote per-DOI statistics to '{per_doi_output_filename}'")
    else:
        print("No per-DOI statistics generated.")

    print("\nStats script finished.")


if __name__ == "__main__":
    main()
