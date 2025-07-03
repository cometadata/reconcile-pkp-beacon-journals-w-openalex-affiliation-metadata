import os
import re
import csv
import sys
import time
import base64
import argparse
import requests
import threading
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_RETRIES = 3
BACKOFF_FACTOR = 0.5
MAX_WORKERS = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fetch HTML from URLs listed in a CSV file, save HTML output using reversible DOI hash for filename, with retries and parallel processing.')
    parser.add_argument('-i', '--input', required=True,
                        help='Path to the input CSV file.')
    parser.add_argument('-l', '--log-file', default="log.log",
                        help='Path to the output log CSV file.')
    parser.add_argument('-d', '--data-dir', required=True,
                        help='Base directory to save downloaded HTML files.')
    parser.add_argument('-w', '--workers', type=int, default=MAX_WORKERS, help=f'Maximum number of concurrent workers (default: {MAX_WORKERS}).')
    return parser.parse_args()


def sanitize_prefix(prefix):
    prefix = prefix.replace('/', '_')
    prefix = re.sub(r'[^\w\-\.]', '_', prefix)
    if prefix.startswith('.') or all(c == '.' for c in prefix):
        prefix = '_' + prefix
    return prefix if prefix else "unknown_sanitized_prefix"


def fetch_html_with_retry(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }
    timeout_seconds = 15
    retries = 0

    while retries <= MAX_RETRIES:
        try:
            response = requests.get(
                url, headers=headers, timeout=timeout_seconds, allow_redirects=True, stream=True)
            response.raise_for_status()

            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' in content_type:
                html_content = response.text
                response.close()
                return 'Success', html_content
            else:
                response.close()
                return 'Failure', f'Skipped - Content-Type is not text/html ({content_type})'

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            retries += 1
            if retries > MAX_RETRIES:
                return 'Failure', f'Error: Max retries exceeded ({type(e).__name__})'
            wait_time = BACKOFF_FACTOR * (2 ** (retries - 1))
            print(f"Retry {retries}/{MAX_RETRIES} for {url} after error: {type(e).__name__}. Waiting {wait_time:.2f}s...")
            time.sleep(wait_time)
        except requests.exceptions.HTTPError as e:
            if 500 <= e.response.status_code < 600:
                retries += 1
                if retries > MAX_RETRIES:
                    return 'Failure', f'Error: Max retries exceeded ({e.response.status_code} {e.response.reason})'
                wait_time = BACKOFF_FACTOR * (2 ** (retries - 1))
                print(f"Retry {retries}/{MAX_RETRIES} for {url} after error: {e.response.status_code}. Waiting {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                return 'Failure', f'Error: HTTP Error: {e.response.status_code} {e.response.reason}'
        except requests.exceptions.RequestException as e:
            if 'response' in locals() and response and not response.raw.closed:
                response.close()
            return 'Failure', f'Error: {type(e).__name__}: {e}'
        except Exception as e:
            if 'response' in locals() and response and not response.raw.closed:
                response.close()
            return 'Failure', f'Error: An unexpected error occurred: {e}'

    return 'Failure', 'Error: Unknown error after retries'


def get_domain(url):
    try:
        parsed_uri = urlparse(url)
        return parsed_uri.netloc if parsed_uri.netloc else parsed_uri.hostname
    except ValueError:
        return None


def worker_task(item_data, domain_lock, base_data_dir):
    url = item_data['url']
    member_id = item_data['member_id']
    prefix = item_data['prefix']
    doi = item_data['doi']
    original_line = item_data['line_number']

    fetch_status, fetch_result = 'Failure', 'Error: Could not acquire domain lock (invalid domain?)'
    output_rel_path = None

    if domain_lock:
        with domain_lock:
            time.sleep(0.1)
            fetch_status, fetch_result = fetch_html_with_retry(url)

    if fetch_status == 'Success':
        html_content = fetch_result
        try:
            safe_prefix = sanitize_prefix(prefix)
            output_dir = os.path.join(
                base_data_dir, str(member_id), safe_prefix)
            os.makedirs(output_dir, exist_ok=True)

            doi_bytes = doi.encode('utf-8')
            encoded_doi_bytes = base64.urlsafe_b64encode(doi_bytes)
            encoded_doi_string = encoded_doi_bytes.decode('ascii')
            output_filename = encoded_doi_string + ".html"

            output_abs_path = os.path.join(output_dir, output_filename)

            with open(output_abs_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            output_rel_path = os.path.relpath(output_abs_path, base_data_dir)
            fetch_details = f"Saved to: {output_rel_path}"

        except IOError as e:
            fetch_status = 'Failure'
            fetch_details = f"Error: Could not write file for DOI {doi}. {e}"
            output_rel_path = None
        except Exception as e:
            fetch_status = 'Failure'
            fetch_details = f"Error: Unexpected error saving file for DOI {doi}. {e}"
            output_rel_path = None
    else:
        fetch_details = fetch_result

    return {
        'url': url,
        'member_id': member_id,
        'prefix': prefix,
        'doi': doi,
        'line_number': original_line,
        'fetch_status': fetch_status,
        'fetch_details': fetch_details,
        'output_rel_path': output_rel_path
    }


def process_csv_parallel(input_filepath, output_log_filepath, base_data_dir, max_workers):
    print(f"Processing CSV file: {input_filepath}")
    print(f"Logging results to: {output_log_filepath}")
    print(f"Saving HTML files to: {base_data_dir}")
    print(f"Using up to {max_workers} workers.")

    items_to_process = []
    required_columns = ['url', 'member_id', 'prefix',
                        'doi', 'line_number']
    try:
        with open(input_filepath, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            missing_cols = [
                col for col in required_columns if col not in reader.fieldnames]
            if missing_cols:
                print(f"Error: Input CSV file '{input_filepath}' is missing required columns: {', '.join(missing_cols)}", file=sys.stderr)
                sys.exit(1)

            for row_num, row in enumerate(reader, start=1):
                item = {col: row.get(col) for col in required_columns}
                if not all(item.values()):
                    print(f"Row {row_num}: Skipped - Missing one or more required values (url, member_id, prefix, doi, line_number)")
                    continue
                items_to_process.append(item)

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_filepath}'", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"Error: Could not read '{input_filepath}'. {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during CSV reading: {e}", file=sys.stderr)
        sys.exit(1)

    domain_locks = {}
    tasks_for_executor = []
    skipped_items = []

    print("Preparing tasks...")
    for item_data in items_to_process:
        url = item_data['url']
        domain = get_domain(url)
        if domain:
            if domain not in domain_locks:
                domain_locks[domain] = threading.Lock()
            tasks_for_executor.append((item_data, domain_locks[domain]))
        else:
            print(f"Skipping invalid URL format: {url} (from line {item_data.get('line_number', 'N/A')})")
            item_data['fetch_status'] = 'Failure'
            item_data['fetch_details'] = 'Error: Invalid URL format'
            item_data['output_rel_path'] = None
            skipped_items.append(item_data)

    results = []
    total_tasks = len(tasks_for_executor)
    completed_tasks = 0

    print(f"Submitting {total_tasks} valid URLs for processing...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(worker_task, item_data, lock, base_data_dir): item_data
            for item_data, lock in tasks_for_executor
        }

        for future in as_completed(future_to_item):
            original_item = future_to_item[future]
            original_url = original_item['url']
            try:
                result_data = future.result()
                results.append(result_data)
                print(f"Processed: {result_data['url']} -> {result_data['fetch_status']}")
                if result_data['fetch_status'] == 'Failure':
                    print(f"  Detail: {result_data['fetch_details']}")

            except Exception as exc:
                print(f'URL {original_url} (line {original_item["line_number"]}) generated an exception in worker: {exc}')
                failure_result = original_item.copy()
                failure_result['fetch_status'] = 'Failure'
                failure_result['fetch_details'] = f'Error: Worker execution failed ({exc})'
                failure_result['output_rel_path'] = None
                results.append(failure_result)

            completed_tasks += 1
            print(f"Progress: {completed_tasks}/{total_tasks} completed.")

    results.extend(skipped_items)

    results.sort(key=lambda x: int(x.get('line_number', 0)))

    print(f"\nWriting {len(results)} results to log file {output_log_filepath}...")
    try:
        with open(output_log_filepath, 'w', newline='', encoding='utf-8') as outfile:
            fieldnames = ['line_number', 'url', 'member_id',
                          'prefix', 'doi', 'fetch_status', 'fetch_details']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for res in results:
                log_details = res.get('fetch_details', 'Unknown Error')
                if res.get('fetch_status') == 'Success' and res.get('output_rel_path'):
                    log_details = f"Saved to: {res['output_rel_path']}"

                writer.writerow({
                    'line_number': res.get('line_number', 'N/A'),
                    'url': res.get('url', ''),
                    'member_id': res.get('member_id', ''),
                    'prefix': res.get('prefix', ''),
                    'doi': res.get('doi', ''),
                    'fetch_status': res.get('fetch_status', 'Failure'),
                    'fetch_details': log_details
                })

    except IOError as e:
        print(f"Error: Could not write log file to '{output_log_filepath}'. {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during log file writing: {e}", file=sys.stderr)
        sys.exit(1)

    success_count = sum(1 for r in results if r['fetch_status'] == 'Success')
    failure_count = len(results) - success_count

    print("\nProcessing complete.")
    print(f"Total input rows considered: {len(items_to_process) + len(skipped_items)}")
    print(f"Total results logged: {len(results)}")
    print(f"Successful HTML saves: {success_count}")
    print(f"Failed/Skipped operations: {failure_count}")


def main():
    args = parse_args()
    try:
        os.makedirs(args.data_dir, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create base data directory '{args.data_dir}'. {e}", file=sys.stderr)
        sys.exit(1)

    process_csv_parallel(args.input, args.log_file,
                         args.data_dir, args.workers)


if __name__ == "__main__":
    main()