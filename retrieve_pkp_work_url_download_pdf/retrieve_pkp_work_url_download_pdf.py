import os
import csv
import argparse
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

# Suppress InsecureRequestWarning for unverified HTTPS requests to allow unverified requests
# from self-signed certificates on some OJS instances.
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


def setup_arg_parser():
    parser = argparse.ArgumentParser(
        description="Process a CSV file to check URLs, download PDFs, and log results.")
    parser.add_argument("-i", "--input", required=True,
                        help="Path to the input CSV file.")
    parser.add_argument(
        "-o", "--output", help="Path to the output CSV file. Defaults to '[input_filename]_processed.csv'.")
    parser.add_argument("-d", "--directory",
                        help="Directory to save downloaded PDF files. Defaults to '[input_filename]_pdfs/'.")
    return parser


def get_url_to_check(row):
    pdf_url = row.get('pdf_url', '').strip()
    if pdf_url:
        return pdf_url
    return row.get('url', '').strip()


def check_url_status(url_to_check):
    if not url_to_check:
        return False, False, "No URL provided"

    try:
        response = requests.head(
            url_to_check, timeout=10, allow_redirects=True, verify=False)

        if response.ok:
            content_type = response.headers.get('Content-Type', '').lower()
            if 'application/pdf' in content_type:
                return True, True, None
            else:
                return True, False, f"Content-Type: {content_type}"
        else:
            return False, False, f"HTTP Status: {response.status_code}"

    except requests.exceptions.RequestException as e:
        return False, False, str(e)


def download_file(url, filepath):
    try:
        response = requests.get(
            url, stream=True, timeout=20, verify=False, allow_redirects=True)
        if response.ok:
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True, None
        else:
            return False, f"Failed to download. HTTP Status: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, str(e)


def extract_filename_from_openalex_id(openalex_id_url):
    if not openalex_id_url or not isinstance(openalex_id_url, str):
        return None
    try:
        name_part = openalex_id_url.strip().split('/')[-1]
        if name_part.startswith('W') and name_part[1:].isdigit():
            return f"{name_part}.pdf"
    except Exception:
        pass
    return None


def process_csv(input_filepath, output_filepath, download_dir):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        print(f"Created download directory: {download_dir}")

    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile, \
                open(output_filepath, 'w',  encoding='utf-8') as outfile:

            reader = csv.DictReader(infile)
            if not reader.fieldnames:
                print(
                    "Error: Could not read header from input CSV.")
                return

            new_fieldnames = reader.fieldnames + \
                ['url_to_check', 'url_resolves', 'url_is_pdf',
                    'downloaded_filename', 'processing_error']
            writer = csv.DictWriter(
                outfile, fieldnames=new_fieldnames)
            writer.writeheader()

            print(f"Processing input file: {input_filepath}")
            print(f"Output will be written to: {output_filepath}")
            print(f"PDFs will be downloaded to: {download_dir}")

            for i, row in enumerate(reader):
                print(f"\nProcessing row {i+1}...")
                output_row = {fieldname: row.get(
                    fieldname, '') for fieldname in reader.fieldnames}

                url_to_check = get_url_to_check(row)
                output_row['url_to_check'] = url_to_check
                output_row['url_resolves'] = False
                output_row['url_is_pdf'] = False
                output_row['downloaded_filename'] = ''
                output_row['processing_error'] = ''

                if not url_to_check:
                    print(f"  No URL found in row.")
                    output_row['processing_error'] = "No URL available in pdf_url or url fields"
                    writer.writerow(output_row)
                    continue

                print(f"  Checking URL: {url_to_check}")
                resolves, is_pdf_by_content_type, err_msg = check_url_status(
                    url_to_check)
                output_row['url_resolves'] = resolves
                output_row['processing_error'] = err_msg if err_msg else ''

                if resolves:
                    print(f"  URL resolves.")
                    openalex_id = row.get('openalex_id', '').strip()
                    filename_base = extract_filename_from_openalex_id(
                        openalex_id)

                    if is_pdf_by_content_type:
                        output_row['url_is_pdf'] = True
                        print(f"  URL content type suggests PDF.")
                        if filename_base:
                            pdf_filepath = os.path.join(
                                download_dir, filename_base)
                            print(f"  Attempting to download PDF to: {pdf_filepath}")
                            download_success, download_error = download_file(
                                url_to_check, pdf_filepath)
                            if download_success:
                                output_row['downloaded_filename'] = filename_base
                                print(f"  Successfully downloaded: {filename_base}")
                            else:
                                print(f"  Failed to download PDF: {download_error}")
                                output_row['processing_error'] = f"Download failed: {download_error}"
                        else:
                            print(f"  Cannot generate filename from OpenAlex ID: {openalex_id}")
                            output_row['processing_error'] = "Cannot generate filename from OpenAlex ID"
                    else:
                        output_row['url_is_pdf'] = False
                        print(f"  URL does not appear to be a PDF based on Content-Type.")
                        if output_row['processing_error']:
                            output_row['processing_error'] += f"; Not PDF based on Content-Type ({err_msg})"
                        else:
                            output_row['processing_error'] = f"Not PDF based on Content-Type ({err_msg})"
                else:
                    print(f"  URL does not resolve or error occurred: {err_msg}")

                writer.writerow(output_row)
            print("\nProcessing complete.")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    download_path = args.downloaddir

    base_filename, ext = os.path.splitext(input_path)

    if not output_path:
        output_path = f"{base_filename}_processed{ext}"

    if not download_path:
        download_path = f"{base_filename}_pdfs"

    process_csv(input_path, output_path, download_path)


if __name__ == "__main__":
    main()
