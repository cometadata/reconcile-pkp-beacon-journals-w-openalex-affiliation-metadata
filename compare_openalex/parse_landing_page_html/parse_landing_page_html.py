import os
import csv
import sys
import base64
import argparse
import datetime
import binascii
from bs4 import BeautifulSoup


def parse_arguments():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(
        description="Extract citation author, institution, and DOI from HTML files (meta tags and ul.authors), merge duplicates, and provide stats."
    )
    parser.add_argument("-i", "--input-dir", required=True,
                        help="Input directory containing HTML files (e.g., 'landing_pages').")
    parser.add_argument("-l", "--log-file", default=f"processing_log_{timestamp}.csv", help="Output CSV file path for logging file processing status. (Default: processing_log_TIMESTAMP.csv)")
    parser.add_argument("-o", "--output-file", default=f"author_affiliation_doi_merged_{timestamp}.csv", help="Output CSV file path for merged author-institution-DOI data. (Default: author_affiliation_doi_merged_TIMESTAMP.csv)")
    parser.add_argument("-s", "--stats-file", default=f"parsing_stats_{timestamp}.csv", help="Output CSV file path for aggregate parsing statistics. (Default: parsing_stats_TIMESTAMP.csv)")
    return parser.parse_args()


def process_html_file(file_path, filename):
    meta_authors_institutions = []
    current_author_meta = None
    current_institutions_meta = []
    has_author_tag_found = False
    has_institution_tag_found = False

    ul_authors_institutions = []

    error_message = None
    decoded_doi = ""
    doi_error_msg = None

    try:
        base_name, _ = os.path.splitext(filename)
        missing_padding = len(base_name) % 4
        if missing_padding:
            base_name += '=' * (4 - missing_padding)
        doi_bytes = base64.b64decode(base_name)
        decoded_doi = doi_bytes.decode('utf-8')
    except (binascii.Error, UnicodeDecodeError) as e:
        doi_error_msg = f"DOI decoding error for filename '{filename}': {e}"
    except Exception as e:
        doi_error_msg = f"Unexpected error decoding DOI for filename '{filename}': {e}"

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f, 'html.parser')

        meta_tags = soup.find_all('meta')
        for tag in meta_tags:
            tag_name = tag.get('name', '').lower()
            content = tag.get('content')

            if not content:
                continue

            content = content.strip()
            if not content:
                continue

            if tag_name == 'citation_author':
                has_author_tag_found = True
                if current_author_meta is not None:
                    meta_authors_institutions.append(
                        (current_author_meta, list(current_institutions_meta), 'meta'))
                current_author_meta = content
                current_institutions_meta = []

            elif tag_name == 'citation_author_institution':
                has_institution_tag_found = True
                if current_author_meta is not None:
                    current_institutions_meta.append(content)

        if current_author_meta is not None:
            meta_authors_institutions.append(
                (current_author_meta, list(current_institutions_meta), 'meta'))

        authors_ul = soup.find('ul', class_='authors')
        if authors_ul:
            author_items = authors_ul.find_all('li', recursive=False)
            for item in author_items:
                author_name_tag = item.find('span', class_='name')
                affiliation_tag = item.find('span', class_='affiliation')

                author_name = author_name_tag.get_text(
                    strip=True) if author_name_tag else None
                affiliation = affiliation_tag.get_text(
                    strip=True) if affiliation_tag else ""

                if author_name:
                    ul_authors_institutions.append(
                        (author_name, affiliation, 'ul.authors'))

    except FileNotFoundError:
        error_message = "File not found"
    except Exception as e:
        error_message = f"Error processing file content: {e}"

    if doi_error_msg:
        if error_message:
            error_message += f"; {doi_error_msg}"
        else:
            error_message = doi_error_msg

    return meta_authors_institutions, ul_authors_institutions, has_author_tag_found, has_institution_tag_found, decoded_doi, error_message


def main():
    args = parse_arguments()
    input_dir = args.input_dir
    log_file_path = args.log_file
    output_file_path = args.output_file
    stats_file_path = args.stats_file

    all_extracted_data = []
    log_data = []

    total_files_found = 0
    files_processed_ok = 0
    files_failed = 0
    files_with_author_tag_present = 0
    files_with_institution_tag_present = 0
    files_with_both_tags_present = 0
    files_with_ul_authors_present = 0

    print(f"Starting scan in directory: {input_dir}")
    print(f"Log file: {log_file_path}")
    print(f"Output file: {output_file_path}")
    print(f"Stats file: {stats_file_path}")

    log_dir = os.path.dirname(log_file_path)
    output_dir = os.path.dirname(output_file_path)
    stats_dir = os.path.dirname(stats_file_path)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if stats_dir:
        os.makedirs(stats_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith('.html'):
                total_files_found += 1
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, start=input_dir)

                meta_pairs_raw, ul_pairs_raw, has_author_tag, has_inst_tag, decoded_doi, error = process_html_file(
                    file_path, filename)

                log_data.append(
                    [relative_path, has_author_tag, has_inst_tag, error or ""])

                processed_current_file = False
                if error:
                    if "DOI decoding error" in error and "Error processing file content" not in error and "File not found" not in error:
                        files_processed_ok += 1
                        processed_current_file = True
                        if has_author_tag:
                            files_with_author_tag_present += 1
                        if has_inst_tag:
                            files_with_institution_tag_present += 1
                        if has_author_tag and has_inst_tag:
                            files_with_both_tags_present += 1
                        if ul_pairs_raw:
                            files_with_ul_authors_present += 1
                    else:
                        files_failed += 1
                        continue
                else:
                    files_processed_ok += 1
                    processed_current_file = True
                    if has_author_tag:
                        files_with_author_tag_present += 1
                    if has_inst_tag:
                        files_with_institution_tag_present += 1
                    if has_author_tag and has_inst_tag:
                        files_with_both_tags_present += 1
                    if ul_pairs_raw:
                        files_with_ul_authors_present += 1

                if processed_current_file:
                    combined_results_for_file = {}

                    for author, institutions, source in meta_pairs_raw:
                        author_stripped = author.strip() if author else ""
                        if not institutions:
                            key = (author_stripped, "")
                            if key not in combined_results_for_file:
                                combined_results_for_file[key] = {
                                    'sources': set(), 'original': [author, ""]}
                            combined_results_for_file[key]['sources'].add(
                                source)
                        else:
                            for institution in institutions:
                                inst_stripped = institution.strip() if institution else ""
                                key = (author_stripped, inst_stripped)
                                if key not in combined_results_for_file:
                                    combined_results_for_file[key] = {
                                        'sources': set(), 'original': [author, institution]}
                                combined_results_for_file[key]['sources'].add(
                                    source)

                    for author, institution, source in ul_pairs_raw:
                        author_stripped = author.strip() if author else ""
                        inst_stripped = institution.strip() if institution else ""
                        key = (author_stripped, inst_stripped)
                        if key not in combined_results_for_file:
                            combined_results_for_file[key] = {
                                'sources': set(), 'original': [author, institution]}
                        combined_results_for_file[key]['sources'].add(source)
                        combined_results_for_file[key]['original'] = [
                            author, institution]

                    for key, data in combined_results_for_file.items():
                        original_author, original_institution = data['original']
                        joined_sources = ";".join(
                            sorted(list(data['sources'])))
                        all_extracted_data.append(
                            [relative_path, decoded_doi, original_author,
                                original_institution, joined_sources]
                        )

    try:
        with open(log_file_path, 'w', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['RelativeFilePath', 'FoundCitationAuthorMetaTag',
                             'FoundCitationAuthorInstitutionMetaTag', 'ProcessingError'])
            writer.writerows(log_data)
        print(f"\nLog file written successfully to: {log_file_path}")
    except IOError as e:
        print(f"\nError writing log file {log_file_path}: {e}", file=sys.stderr)

    try:
        with open(output_file_path, 'w', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['RelativeFilePath', 'DOI',
                             'Author', 'Institution', 'Source'])
            writer.writerows(all_extracted_data)
        print(f"Extracted data written successfully to: {output_file_path}")
    except IOError as e:
        print(f"Error writing output file {output_file_path}: {e}", file=sys.stderr)

    print("\n--- Parsing Statistics ---")
    print(f"Total HTML files found: {total_files_found}")
    print(f"Files successfully processed (content parsed): {files_processed_ok}")
    print(f"Files failed processing (content/not found): {files_failed}")

    stats_list_for_csv = [
        ['Statistic', 'Value'],
        ['Total HTML files found', total_files_found],
        ['Files successfully processed (content parsed)', files_processed_ok],
        ['Files failed processing (content/not found)', files_failed]
    ]

    if files_processed_ok > 0:
        perc_author_tag = (files_with_author_tag_present /
                           files_processed_ok) * 100
        perc_inst_tag = (files_with_institution_tag_present /
                         files_processed_ok) * 100
        perc_both_tags = (files_with_both_tags_present /
                          files_processed_ok) * 100
        perc_ul_authors = (files_with_ul_authors_present /
                           files_processed_ok) * 100

        print(f"Files containing citation_author meta tag: {files_with_author_tag_present} ({perc_author_tag:.2f}%)")
        print(f"Files containing citation_author_institution meta tag: {files_with_institution_tag_present} ({perc_inst_tag:.2f}%)")
        print(f"Files containing both meta tags: {files_with_both_tags_present} ({perc_both_tags:.2f}%)")
        print(f"Files containing ul.authors structure: {files_with_ul_authors_present} ({perc_ul_authors:.2f}%)")

        stats_list_for_csv.extend([
            ['Files containing citation_author meta tag',
                files_with_author_tag_present],
            ['Files containing citation_author meta tag (%)', f"{perc_author_tag:.2f}"],
            ['Files containing citation_author_institution meta tag',
             files_with_institution_tag_present],
            ['Files containing citation_author_institution meta tag (%)', f"{perc_inst_tag:.2f}"],
            ['Files containing both meta tags', files_with_both_tags_present],
            ['Files containing both meta tags (%)', f"{perc_both_tags:.2f}"],
            ['Files containing ul.authors structure',
             files_with_ul_authors_present],
            ['Files containing ul.authors structure (%)', f"{perc_ul_authors:.2f}"]
        ])
    else:
        print("No files were processed successfully, cannot calculate percentages for tag/structure presence.")
        stats_list_for_csv.extend([
            ['Files containing citation_author meta tag',
                files_with_author_tag_present],
            ['Files containing citation_author meta tag (%)', 'N/A'],
            ['Files containing citation_author_institution meta tag',
             files_with_institution_tag_present],
            ['Files containing citation_author_institution meta tag (%)', 'N/A'],
            ['Files containing both meta tags', files_with_both_tags_present],
            ['Files containing both meta tags (%)', 'N/A'],
            ['Files containing ul.authors structure',
             files_with_ul_authors_present],
            ['Files containing ul.authors structure (%)', 'N/A']
        ])

    total_rows = len(all_extracted_data)
    print(f"\nTotal unique DOI/author/institution rows extracted (merged): {total_rows}")
    stats_list_for_csv.append(
        ['Total unique DOI/author/institution rows extracted (merged)', total_rows])

    if all_extracted_data:
        unique_authors = set(row[2] for row in all_extracted_data)
        unique_institutions = set(row[3]
                                  for row in all_extracted_data if row[3])
        valid_pairs_count = sum(1 for row in all_extracted_data if row[3])
        authors_without_inst = sum(
            1 for row in all_extracted_data if not row[3])
        rows_from_meta_only = sum(
            1 for row in all_extracted_data if row[4] == 'meta')
        rows_from_ul_only = sum(
            1 for row in all_extracted_data if row[4] == 'ul.authors')
        rows_from_both = sum(
            1 for row in all_extracted_data if 'meta' in row[4] and 'ul.authors' in row[4])

        print(f"Total valid author-institution pairs extracted (non-empty institution): {valid_pairs_count}")
        print(f"Author rows with missing/empty institution: {authors_without_inst}")
        print(f"Rows sourced ONLY from meta tags: {rows_from_meta_only}")
        print(f"Rows sourced ONLY from ul.authors: {rows_from_ul_only}")
        print(f"Rows sourced from BOTH meta and ul.authors: {rows_from_both}")
        print(f"Unique authors found (merged): {len(unique_authors)}")
        print(f"Unique institutions found (non-empty, merged): {len(unique_institutions)}")

        files_logged_with_inst_meta_tags = set(
            log_row[0] for log_row in log_data if log_row[2])
        authors_missing_inst_in_meta_tagged_files = sum(1 for data_row in all_extracted_data
                                                        if not data_row[3] and data_row[0] in files_logged_with_inst_meta_tags)

        print(f"Author rows missing institution FROM FILES REPORTED HAVING INSTITUTION META TAGS: {authors_missing_inst_in_meta_tagged_files}")

        stats_list_for_csv.extend([
            ['Total valid author-institution pairs extracted (non-empty institution)', valid_pairs_count],
            ['Author rows with missing/empty institution', authors_without_inst],
            ['Rows sourced ONLY from meta tags', rows_from_meta_only],
            ['Rows sourced ONLY from ul.authors', rows_from_ul_only],
            ['Rows sourced from BOTH meta and ul.authors', rows_from_both],
            ['Unique authors found (merged)', len(unique_authors)],
            ['Unique institutions found (non-empty, merged)',
             len(unique_institutions)],
            ['Author rows missing institution from files with institution meta tags',
                authors_missing_inst_in_meta_tagged_files]
        ])
    else:
        print("No DOI/author/institution data was extracted.")
        stats_list_for_csv.extend([
            ['Total valid author-institution pairs extracted (non-empty institution)', 0],
            ['Author rows with missing/empty institution', 0],
            ['Rows sourced ONLY from meta tags', 0],
            ['Rows sourced ONLY from ul.authors', 0],
            ['Rows sourced from BOTH meta and ul.authors', 0],
            ['Unique authors found (merged)', 0],
            ['Unique institutions found (non-empty, merged)', 0],
            ['Author rows missing institution from files with institution meta tags', 0]
        ])

    try:
        with open(stats_file_path, 'w', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(stats_list_for_csv)
        print(f"\nStatistics written successfully to: {stats_file_path}")
    except IOError as e:
        print(f"\nError writing stats file {stats_file_path}: {e}", file=sys.stderr)

    print("\nScript finished.")


if __name__ == "__main__":
    main()
