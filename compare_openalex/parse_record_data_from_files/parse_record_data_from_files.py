import re
import csv
import sys
import gzip
import json
import logging
import argparse
from pathlib import Path
from urllib.parse import urlparse, unquote
from datetime import datetime, timezone
from tqdm import tqdm

LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
DEFAULT_OUTPUT_CSV = "extracted_data.csv"
CSV_HEADERS = [
    "member_id",
    "prefix",
    "doi",
    "url",
    "pdf_url",
    "deposit_timestamp_yyyymmddhhmmss",
    "openalex_id",
    "source_file_relative",
    "line_number",
    "parse_status",
    "parse_error_details"
]

DOI_PREFIX_REGEX = re.compile(r"^(?:https?://doi\.org/)?(10\.[0-9]+)(?:/|$)")


def setup_logging(log_level_str):
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format=LOG_FORMAT,
                        datefmt='%Y-%m-%d %H:%M:%S', force=True)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Parses gzipped JSONL files (Crossref or OpenAlex) to extract metadata into a single CSV.")
    parser.add_argument(
        "-i", "--input-dir", required=True, help="Base directory containing data files (e.g., data.jsonl.gz). Structure might matter for Crossref.")
    parser.add_argument(
        "-s", "--source-type", required=True, choices=["crossref", "openalex"], help="Type of source files being processed.")
    parser.add_argument(
        "-o", "--output-csv", default=DEFAULT_OUTPUT_CSV, help=f"Path for the output CSV file (default: {DEFAULT_OUTPUT_CSV}).")
    parser.add_argument(
        "-l", "--log-level", default="INFO", choices=[
            "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set console logging level (default: INFO).")
    parser.add_argument(
        "-m", "--max-records-per-file", type=int, default=None,
        help="Process only the first N records per JSONL file (for testing).")
    args = parser.parse_args()
    args.input_dir = Path(args.input_dir)
    args.output_csv = Path(args.output_csv)
    return args


def find_jsonl_gz_files(input_dir):
    input_dir_path = Path(input_dir)
    if not input_dir_path.is_dir():
        logging.critical(f"Input directory not found or is not a directory: {input_dir_path}")
        return None
    logging.info(f"Searching for '*.jsonl.gz' files in {input_dir_path}...")
    files = list(input_dir_path.rglob("*.jsonl.gz"))
    logging.info(f"Found {len(files)} '*.jsonl.gz' files.")
    return files


def get_relative_path_or_str(path, base):
    try:
        resolved_path = Path(path).resolve()
        resolved_base = Path(base).resolve()
        if resolved_base in resolved_path.parents or resolved_base == resolved_path.parent:
            return str(resolved_path.relative_to(resolved_base))
        else:
            logging.debug(f"Path {resolved_path} is not directly relative to base {resolved_base}. Using full path string.")
            return str(resolved_path)
    except (ValueError, Exception) as e:
        logging.warning(f"Could not determine relative path for {path} against base {base}: {e}. Using full path string.")
        return str(path)


def is_valid_url(url_str):
    if not url_str or not isinstance(url_str, str):
        return False, "URL is missing or not a string"
    try:
        parsed = urlparse(url_str)
        if parsed.scheme and parsed.netloc:
            return True, ""
        if not parsed.scheme and parsed.netloc:
            fixed_url = f"http://{url_str}"
            parsed_fixed = urlparse(fixed_url)
            if parsed_fixed.scheme and parsed_fixed.netloc:
                logging.debug(f"Fixed missing URL scheme for {url_str} -> {fixed_url}")
                return True, fixed_url
            else:
                return False, f"Invalid structure (even after adding http://): {url_str}"
        else:
            return False, f"Invalid structure (missing scheme or netloc): {url_str}"
    except ValueError:
        return False, f"Parsing error for URL: {url_str}"


def extract_prefix_from_doi(doi_str):
    if not doi_str or not isinstance(doi_str, str):
        return None
    decoded_doi = unquote(doi_str)
    match = DOI_PREFIX_REGEX.match(decoded_doi)
    if match:
        return match.group(1)
    logging.debug(f"Could not extract prefix from DOI string: {doi_str}")
    return None


def extract_line_data_crossref(line_number, line_content):
    url, doi, deposit_timestamp_str = None, None, None
    status = "OK"
    error_details = ""
    openalex_id = None
    pdf_url = None

    try:
        line_stripped = line_content.strip()
        if not line_stripped:
            return None, None, None, None, None, "EMPTY_LINE", ""

        record = json.loads(line_stripped)

        url_str = record.get("resource", {}).get("primary", {}).get("URL")
        is_valid, url_info = is_valid_url(url_str)
        if is_valid:
            url = url_info if isinstance(
                url_info, str) and url_info.startswith("http") else url_str
        elif url_str:
            status = "URL_INVALID"
            error_details = url_info
            logging.warning(f"Line {line_number}: {status} - {error_details}")
        else:
            status = "URL_MISSING"
            error_details = "Primary resource URL not found in record."
            logging.debug(f"Line {line_number}: {status}")

        doi_str = record.get("DOI")
        if doi_str and isinstance(doi_str, str) and doi_str.strip():
            doi = doi_str.strip()
        else:
            logging.debug(f"Line {line_number}: DOI missing or invalid type. URL='{url}'")

        deposited_ms = record.get("deposited", {}).get("timestamp")
        if deposited_ms and isinstance(deposited_ms, (int, float)):
            try:
                ts_seconds = deposited_ms / 1000.0
                if ts_seconds > 0:
                    deposited_dt = datetime.fromtimestamp(
                        ts_seconds, tz=timezone.utc)
                    if 1970 <= deposited_dt.year <= datetime.now(timezone.utc).year + 5:
                        deposit_timestamp_str = deposited_dt.strftime(
                            '%Y%m%d%H%M%S')
                    else:
                        raise ValueError(f"Timestamp {deposited_ms} resulted in unreasonable year {deposited_dt.year}")
                else:
                    raise ValueError(f"Non-positive timestamp value: {deposited_ms}")

            except (ValueError, OSError, TypeError) as dt_err:
                if status == "OK":
                    status = "TIMESTAMP_FORMAT_ERROR"
                error_details += f" | Could not format deposit timestamp {deposited_ms}: {dt_err}"
                logging.warning(f"Line {line_number}: Timestamp error for DOI {doi}, URL {url}: {dt_err}")
        else:
            logging.debug(f"Line {line_number}: Deposit timestamp missing or invalid type. DOI='{doi}', URL='{url}'")

    except json.JSONDecodeError as e:
        status = "JSON_DECODE_ERROR"
        error_details = f"Invalid JSON: {e} - Line snippet: {line_content[:100]}..."
        logging.warning(f"Line {line_number}: {status} - {error_details}")
        url, doi, deposit_timestamp_str, openalex_id, pdf_url = None, None, None, None, None
    except Exception as e:
        status = "UNEXPECTED_PARSE_ERROR"
        error_details = f"Unexpected error parsing line ({type(e).__name__}): {e} - Line snippet: {line_content[:100]}..."
        logging.error(f"Line {line_number}: {status} - {error_details}", exc_info=logging.getLogger().isEnabledFor(logging.DEBUG))
        url, doi, deposit_timestamp_str, openalex_id, pdf_url = None, None, None, None, None

    if status == "OK" and not url:
        logging.debug(f"Line {line_number}: Parsed OK but primary URL is missing.")
    return url, doi, deposit_timestamp_str, openalex_id, pdf_url, status, error_details


def extract_line_data_openalex(line_number, line_content):
    url, pdf_url, doi = None, None, None
    member_id = None
    deposit_timestamp_str = None
    prefix = None
    openalex_id = None
    status = "OK"
    error_details = ""

    try:
        line_stripped = line_content.strip()
        if not line_stripped:
            return member_id, prefix, doi, url, pdf_url, deposit_timestamp_str, openalex_id, "EMPTY_LINE", ""

        record = json.loads(line_stripped)
        if not isinstance(record, dict):
            status = "JSON_INVALID_STRUCTURE"
            error_details = "Parsed JSON is not a dictionary/object."
            logging.warning(f"Line {line_number}: {status} - {error_details}")
            return member_id, prefix, doi, url, pdf_url, deposit_timestamp_str, openalex_id, status, error_details

        openalex_id = record.get("id")
        if not openalex_id:
            logging.debug(f"Line {line_number}: OpenAlex ID ('id' field) missing.")
        elif not isinstance(openalex_id, str):
            logging.warning(f"Line {line_number}: OpenAlex ID ('id' field) is not a string: {type(openalex_id)}. Value: {openalex_id}")
            if status == "OK":
                status = "ID_INVALID_TYPE"
            error_details += f" | OpenAlex ID field is not a string: {type(openalex_id)}"

        url_str = None
        primary_loc = record.get("primary_location")
        if primary_loc and isinstance(primary_loc, dict):
            url_str = primary_loc.get("landing_page_url")

        if not url_str:
            best_oa_loc = record.get("best_oa_location")
            if best_oa_loc and isinstance(best_oa_loc, dict):
                url_str_best_oa = best_oa_loc.get("landing_page_url")
                if url_str_best_oa:
                    url_str = url_str_best_oa
                    logging.debug(f"Line {line_number}: Used best_oa_location landing_page_url as primary was missing/invalid.")

        is_valid, url_info = is_valid_url(url_str)
        if is_valid:
            url = url_info if isinstance(
                url_info, str) and url_info.startswith("http") else url_str
        elif url_str:
            if status == "OK":
                status = "URL_INVALID"
            error_details += f" | Landing Page URL Invalid: {url_info}"
            logging.warning(f"Line {line_number}: URL_INVALID - {url_info}")
        else:
            if status == "OK":
                status = "URL_MISSING"
            error_details += " | Primary and best_oa landing page URL not found or invalid in record."
            logging.debug(f"Line {line_number}: URL_MISSING")

        pdf_url_str = None
        if primary_loc and isinstance(primary_loc, dict):
            pdf_url_str = primary_loc.get("pdf_url")

        if not pdf_url_str:
            best_oa_loc = record.get("best_oa_location")
            if best_oa_loc and isinstance(best_oa_loc, dict):
                pdf_url_str_best_oa = best_oa_loc.get("pdf_url")
                if pdf_url_str_best_oa:
                    pdf_url_str = pdf_url_str_best_oa
                    logging.debug(f"Line {line_number}: Used best_oa_location pdf_url as primary was missing/invalid.")

        is_pdf_url_valid, pdf_url_info = is_valid_url(pdf_url_str)
        if is_pdf_url_valid:
            pdf_url = pdf_url_info if isinstance(
                pdf_url_info, str) and pdf_url_info.startswith("http") else pdf_url_str
        elif pdf_url_str:
            logging.warning(f"Line {line_number}: PDF_URL_INVALID - {pdf_url_info}. Landing URL: {url}")
            if status == "OK":
                status = "PDF_URL_INVALID"
            error_details += f" | PDF URL Invalid: {pdf_url_info}"
        else:
            logging.debug(f"Line {line_number}: PDF_URL_MISSING. Landing URL: {url}")

        doi_str = record.get("doi")

        if not doi_str and primary_loc and isinstance(primary_loc, dict):
            doi_str = primary_loc.get("doi")
            if doi_str:
                logging.debug(f"Line {line_number}: Used DOI from primary_location.")

        if not doi_str and record.get("best_oa_location") and isinstance(record.get("best_oa_location"), dict):
            doi_str = record.get("best_oa_location").get("doi")
            if doi_str:
                logging.debug(f"Line {line_number}: Used DOI from best_oa_location.")

        if doi_str and isinstance(doi_str, str) and doi_str.strip():
            doi = doi_str.strip()
            prefix = extract_prefix_from_doi(doi)
            if not prefix:
                logging.debug(f"Line {line_number}: Could not extract prefix from DOI '{doi}'. URL='{url}'")
        else:
            logging.debug(f"Line {line_number}: DOI missing or invalid type after checking all locations. URL='{url}'")

        deposit_timestamp_str = None
        member_id = None

    except json.JSONDecodeError as e:
        status = "JSON_DECODE_ERROR"
        error_details = f"Invalid JSON: {e} - Line snippet: {line_content[:100]}..."
        logging.warning(f"Line {line_number}: {status} - {error_details}")
        url, pdf_url, doi, prefix, openalex_id = None, None, None, None, None
    except Exception as e:
        status = "UNEXPECTED_PARSE_ERROR"
        error_details = f"Unexpected error parsing line ({type(e).__name__}): {e} - Line snippet: {line_content[:100]}..."
        logging.error(f"Line {line_number}: {status} - {error_details}", exc_info=logging.getLogger().isEnabledFor(logging.DEBUG))
        url, pdf_url, doi, prefix, openalex_id = None, None, None, None, None

    if status == "OK" and not url:
        logging.debug(f"Line {line_number}: Parsed OK but final landing page URL is missing.")
    if status == "OK" and not pdf_url:
        logging.debug(f"Line {line_number}: Parsed OK but final PDF URL is missing.")
    if status == "OK" and not openalex_id:
        logging.debug(f"Line {line_number}: Parsed OK but OpenAlex ID is missing.")

    return member_id, prefix, doi, url, pdf_url, deposit_timestamp_str, openalex_id, status, error_details.strip(" | ")


def main():
    args = parse_arguments()
    setup_logging(args.log_level)

    input_dir_path = args.input_dir.resolve()
    output_csv_path = args.output_csv
    source_type = args.source_type

    logging.info(f"--- Starting {source_type.capitalize()} Metadata Extraction ---")
    logging.info(f"Input Directory: {input_dir_path}")
    logging.info(f"Output CSV File: {output_csv_path}")
    logging.info(f"Source Type: {source_type}")
    if args.max_records_per_file:
        logging.info(f"Max records per file: {args.max_records_per_file}")

    jsonl_files = find_jsonl_gz_files(input_dir_path)
    if jsonl_files is None:
        return
    if not jsonl_files:
        logging.warning("No input '*.jsonl.gz' files found. Exiting.")
        return

    processed_files_count = 0
    processed_lines_total = 0
    extracted_records_count = 0
    error_records_count = 0
    missing_doi_count = 0
    missing_timestamp_count = 0
    missing_url_count = 0
    missing_pdf_url_count = 0
    missing_openalex_id_count = 0

    try:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_csv_path, 'w', encoding='utf-8') as f_in:
            writer = csv.DictWriter(f_in, fieldnames=CSV_HEADERS)
            writer.writeheader()
            logging.info(f"Opened output CSV: {output_csv_path}")

            for file_path_obj in tqdm(jsonl_files, desc="Processing Files"):
                processed_files_count += 1
                resolved_file_path = file_path_obj.resolve()
                relative_file_path_str = get_relative_path_or_str(
                    resolved_file_path, input_dir_path)

                path_member_id = None
                path_prefix = None
                if source_type == "crossref":
                    try:
                        relative_parts = Path(relative_file_path_str).parts
                        if len(relative_parts) >= 3 and relative_parts[-1].endswith('.jsonl.gz'):
                            path_member_id = relative_parts[-3]
                            path_prefix = relative_parts[-2]

                            if not path_member_id.isdigit():
                                logging.warning(f"Expected numeric Member ID from path, got '{path_member_id}' in {relative_file_path_str}")
                                path_member_id = f"INVALID_PATH_MEMBER ({path_member_id})"
                            if not path_prefix.startswith("10.") or not (len(path_prefix.split('.')) > 1 and path_prefix.split('.')[1].isdigit()):
                                logging.warning(f"Expected DOI prefix (10.xxx) from path, got '{path_prefix}' in {relative_file_path_str}")
                                path_prefix = f"INVALID_PATH_PREFIX ({path_prefix})"

                        else:
                            logging.warning(f"Could not determine Crossref member/prefix from path structure: {relative_file_path_str}. Path parsing skipped.")
                    except Exception as e:
                        logging.error(f"Error determining path structure for {relative_file_path_str}: {e}")

                logging.debug(f"Processing file: {relative_file_path_str}")

                try:
                    with gzip.open(resolved_file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                        processed_lines_file = 0
                        for i, line in enumerate(f):
                            line_num = i + 1
                            processed_lines_total += 1
                            processed_lines_file += 1

                            member_id, prefix, doi, url, pdf_url, deposit_ts, openalex_id = None, None, None, None, None, None, None
                            status, error_details = "INIT_ERROR", "Initialization failed"

                            try:
                                if source_type == "crossref":
                                    url, doi, deposit_ts, _, pdf_url_cr, status, error_details = extract_line_data_crossref(
                                        line_num, line)
                                    member_id = path_member_id
                                    prefix = path_prefix
                                    if not prefix and doi:
                                        prefix = extract_prefix_from_doi(doi)

                                elif source_type == "openalex":
                                    member_id, prefix, doi, url, pdf_url, deposit_ts, openalex_id, status, error_details = extract_line_data_openalex(
                                        line_num, line)
                                else:
                                    status = "CONFIG_ERROR"
                                    error_details = f"Unknown source_type: {source_type}"
                                    logging.critical(error_details)

                            except Exception as parse_ex:
                                status = "FATAL_PARSE_ERROR"
                                error_details = f"Unhandled exception in parser function for line: {parse_ex}"
                                logging.error(f"Line {line_num} in {relative_file_path_str}: {status} - {error_details}", exc_info=True)

                            if status == "EMPTY_LINE":
                                processed_lines_total -= 1
                                continue

                            row_data = {
                                "member_id": member_id if member_id else "",
                                "prefix": prefix if prefix else "",
                                "doi": doi if doi else "",
                                "url": url if url else "",
                                "pdf_url": pdf_url if pdf_url else "",
                                "deposit_timestamp_yyyymmddhhmmss": deposit_ts if deposit_ts else "",
                                "openalex_id": openalex_id if openalex_id else "",
                                "source_file_relative": relative_file_path_str,
                                "line_number": line_num,
                                "parse_status": status,
                                "parse_error_details": error_details if error_details else ""
                            }
                            writer.writerow(row_data)

                            if status == "OK" or status == "PDF_URL_INVALID" or status == "URL_INVALID" or status == "ID_INVALID_TYPE":
                                extracted_records_count += 1
                                if not doi:
                                    missing_doi_count += 1
                                if not url:
                                    missing_url_count += 1

                                if source_type == "crossref":
                                    if not deposit_ts:
                                        missing_timestamp_count += 1
                                elif source_type == "openalex":
                                    if not openalex_id:
                                        missing_openalex_id_count += 1
                                    if not pdf_url:
                                        missing_pdf_url_count += 1
                            elif status != "EMPTY_LINE":
                                error_records_count += 1

                            if args.max_records_per_file and processed_lines_file >= args.max_records_per_file:
                                logging.debug(f"Reached max records ({args.max_records_per_file}) for {relative_file_path_str}")
                                break

                except (gzip.BadGzipFile, FileNotFoundError, EOFError, IOError) as e:
                    logging.error(f"Failed to read or process file {relative_file_path_str}: {e}")
                    error_records_count += 1
                    continue
                except Exception as e:
                    logging.error(f"An unexpected error occurred while processing file {relative_file_path_str}: {e}", exc_info=True)
                    error_records_count += 1
                    continue

    except IOError as e:
        logging.critical(f"Could not write to output CSV file {output_csv_path}: {e}")
        return
    except Exception as e:
        logging.critical(f"An unexpected error occurred during processing: {e}", exc_info=True)
        return

    logging.info("--- Extraction Finished ---")
    logging.info(f"Processed {processed_files_count} files.")
    logging.info(f"Processed {processed_lines_total} total lines (excluding fully empty lines).")
    logging.info(f"Successfully parsed records (status='OK' or minor issues like 'URL_INVALID'): {extracted_records_count}")
    logging.info(f"Records with critical parsing errors (status not 'OK' or minor): {error_records_count}")

    logging.info(f"--- Statistics for Parsed Records ({extracted_records_count} total) ---")
    logging.info(f"Missing DOI: {missing_doi_count}")
    logging.info(f"Missing Landing Page URL: {missing_url_count}")

    if source_type == "crossref":
        logging.info(f"Missing Deposit Timestamp: {missing_timestamp_count}")
        logging.info(f"Missing PDF URL: N/A for Crossref (not extracted)")
        logging.info(f"Missing OpenAlex ID: N/A for Crossref (not extracted)")
    elif source_type == "openalex":
        logging.info(f"Missing PDF URL: {missing_pdf_url_count}")
        logging.info(f"Missing OpenAlex ID: {missing_openalex_id_count}")
        logging.info(f"Missing Deposit Timestamp: N/A for OpenAlex (not available)")

    logging.info(f"Output saved to: {output_csv_path}")


if __name__ == "__main__":
    main()
