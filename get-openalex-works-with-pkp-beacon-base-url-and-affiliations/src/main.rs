use std::{
    collections::{HashMap, HashSet, VecDeque},
    fs::{self, File, OpenOptions},
    io::{self, BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use csv;
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use glob::glob;
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, error, info, warn, LevelFilter};
use rayon::prelude::*;
use serde::Deserialize;
use simple_logger::SimpleLogger;
use time::macros::format_description;
use url::Url;

#[cfg(target_os = "linux")]
use std::fs::read_to_string;
#[cfg(target_os = "windows")]
use std::process::Command as WinCommand;


#[derive(Parser)]
#[command(name = "OpenAlex URL/Affiliation Filter")]
#[command(about = "Filters OpenAlex JSONL.gz files based on locations[].landing_page_url matching ANY URL in a CSV and *at least one non-empty* authorships[].raw_affiliation_strings, organizing by DOI prefix.")]
#[command(version = "2.1.0")]
struct Cli {
    #[arg(short, long, help = "Directory containing input JSONL.gz files", required = true)]
    input_dir: String,

    #[arg(short, long, help = "Base directory for organized output structure (prefix/data.jsonl.gz)", required = true)]
    output_dir: String,

    #[arg(long, short = 'b', help = "Path to CSV file containing base URLs (header 'base_url')", required = true)]
    base_urls_csv: String,

    #[arg(long, default_value = "256", help = "Max open final prefix output files")]
    max_open_prefix_files: usize,

    #[arg(short, long, default_value = "INFO", help = "Logging level (DEBUG, INFO, WARN, ERROR)")]
    log_level: String,

    #[arg(short, long, default_value = "0", help = "Number of threads to use (0 for auto)")]
    threads: usize,

    #[arg(short, long, default_value = "60", help = "Interval in seconds to log statistics")]
    stats_interval: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct DoiPrefix(String);

#[derive(Deserialize, Debug)]
struct OpenAlexRecord {
    doi: Option<String>,
    authorships: Option<Vec<Authorship>>,
    locations: Option<Vec<Location>>,
}

#[derive(Deserialize, Debug)]
struct Authorship {
    raw_affiliation_strings: Option<Vec<String>>,
}

#[derive(Deserialize, Debug)]
struct Location {
    landing_page_url: Option<String>,
}

#[derive(Default)]
struct Stats {
    lines_read: AtomicU64,
    json_parse_errors: AtomicU64,
    url_location_missing: AtomicU64,
    url_landing_page_missing: AtomicU64,
    url_landing_page_parse_errors: AtomicU64,
    url_no_match: AtomicU64,
    affiliation_author_missing: AtomicU64,
    affiliation_raw_missing: AtomicU64,
    affiliation_all_empty_or_missing: AtomicU64,
    doi_missing: AtomicU64,
    lines_passed_filter: AtomicU64,
    prefix_files_opened: AtomicU64,
    unique_prefixes_written: Mutex<HashSet<DoiPrefix>>,
}

impl Stats {
    fn new() -> Self {
        Default::default()
    }

    fn log_current_stats(&self, stage: &str) {
        let lines_read = self.lines_read.load(Ordering::Relaxed);
        let json_err = self.json_parse_errors.load(Ordering::Relaxed);
        let url_loc_miss = self.url_location_missing.load(Ordering::Relaxed);
        let url_lp_miss = self.url_landing_page_missing.load(Ordering::Relaxed);
        let url_lp_parse_err = self.url_landing_page_parse_errors.load(Ordering::Relaxed);
        let url_no_match = self.url_no_match.load(Ordering::Relaxed);
        let aff_auth_miss = self.affiliation_author_missing.load(Ordering::Relaxed);
        let aff_raw_miss = self.affiliation_raw_missing.load(Ordering::Relaxed);
        let aff_all_empty = self.affiliation_all_empty_or_missing.load(Ordering::Relaxed);
        let doi_miss = self.doi_missing.load(Ordering::Relaxed);
        let passed = self.lines_passed_filter.load(Ordering::Relaxed);
        let files_opened = self.prefix_files_opened.load(Ordering::Relaxed);
        let unique_prefixes = self.unique_prefixes_written.lock().unwrap().len();

        info!("--- Periodic Stats ({}) ---", stage);
        info!(" Lines Read (Input): {}", lines_read);
        info!(" Lines Written (Final Output): {}", passed);
        info!(" Lines Filtered Out:");
        info!("    URL Related:");
        info!("       No/Empty 'locations': {}", url_loc_miss);
        info!("       No 'landing_page_url' found (in any location): {}", url_lp_miss);
        info!("       'landing_page_url' parse errors: {}", url_lp_parse_err);
        info!("       No 'landing_page_url' matched base URLs: {}", url_no_match);
        info!("    Affiliation Related (URL matched but affiliation check failed):");
        info!("       No/Empty 'authorships': {}", aff_auth_miss);
        info!("       'raw_affiliation_strings' all empty or missing: {}", aff_all_empty);
        info!(" Lines Discarded (Pre-Filter):");
        info!("    JSON Parse Errors: {}", json_err);
        info!(" Records Passed Filter but Missing/Invalid DOI: {}", doi_miss);
        info!(" Metrics:");
        info!("    Unique Prefixes Written (so far): {}", unique_prefixes);
        info!("    Prefix Output Files Opened (cumulative): {}", files_opened);
        info!("    Occurrences of missing 'raw_affiliation_strings' field (across all checked authorships): {}", aff_raw_miss);
        info!("------------------------------");
    }
}


fn find_gz_files_excluding_csv_gz<P: AsRef<Path>>(directory: P) -> Result<Vec<PathBuf>> {
    let pattern = directory.as_ref().join("**/*.gz");
    let pattern_str = pattern.to_string_lossy();
    info!("Searching for files matching pattern: {}", pattern_str);
    let paths: Vec<PathBuf> = glob(&pattern_str)?
        .filter_map(Result::ok)
        .filter(|path| path.is_file())
        .filter(|path| {
            if let Some(filename_osstr) = path.file_name() {
                if let Some(filename_str) = filename_osstr.to_str() {
                    return !filename_str.ends_with(".csv.gz");
                }
            }
            false
        })
        .collect();

    if paths.is_empty() {
        warn!(
            "No .gz files found (excluding .csv.gz) matching the pattern: {}",
            pattern_str
        );
    }
    Ok(paths)
}

fn read_base_urls_from_csv<P: AsRef<Path>>(path: P) -> Result<Vec<Url>> {
    let file_path_display = path.as_ref().display().to_string();
    let file = File::open(path.as_ref())
        .with_context(|| format!("Failed to open base URLs CSV file: {}", file_path_display))?;
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    let headers = rdr.headers()?.clone();
    if headers.get(0) != Some("base_url") {
        return Err(anyhow!("Expected header 'base_url' not found in {}", file_path_display));
    }

    let mut base_urls = Vec::new();
    let mut line_num = 1;

    for result in rdr.records() {
        line_num += 1;
        let record = result.with_context(|| format!("Failed to read record line {} from {}", line_num, file_path_display))?;
        if let Some(url_val) = record.get(0) {
            let trimmed_url = url_val.trim();
            if !trimmed_url.is_empty() {
                match Url::parse(trimmed_url) {
                    Ok(parsed_url) => {
                        base_urls.push(parsed_url);
                    }
                    Err(e) => {
                        warn!(
                            "Skipping invalid URL '{}' on line {} in {}: {}",
                            trimmed_url, line_num, file_path_display, e
                        );
                    }
                }
            }
        }
    }

    if base_urls.is_empty() {
        Err(anyhow!("No valid base_url values found or parsed in {}", file_path_display))
    } else {
        info!("Successfully parsed {} base URLs from {}", base_urls.len(), file_path_display);
        Ok(base_urls)
    }
}


fn check_location_url_match_any(locations: Option<&Vec<Location>>, base_urls: &[Url], stats: &Stats) -> bool {
    let locations_vec = match locations {
        Some(locs) if !locs.is_empty() => locs,
        _ => {
            stats.url_location_missing.fetch_add(1, Ordering::Relaxed);
            return false;
        }
    };

    let mut found_any_landing_page = false;
    let mut found_match = false;

    for location in locations_vec {
        if let Some(landing_url_str) = location.landing_page_url.as_deref() {
            found_any_landing_page = true;
            match Url::parse(landing_url_str) {
                Ok(landing_url) => {
                    let record_scheme = landing_url.scheme();
                    let record_host = landing_url.host_str();
                    let record_port = landing_url.port_or_known_default();

                    for base_url in base_urls {
                        let scheme_match = record_scheme == base_url.scheme();
                        let host_match = record_host == base_url.host_str();
                        let port_match = record_port == base_url.port_or_known_default();

                        if scheme_match && host_match && port_match {
                            found_match = true;
                            break;
                        }
                    }
                }
                Err(_e) => {
                    stats.url_landing_page_parse_errors.fetch_add(1, Ordering::Relaxed);
                }
            }
            if found_match {
                break;
            }
        }
    }

    if found_match {
        true
    } else {
        if !found_any_landing_page {
            stats.url_landing_page_missing.fetch_add(1, Ordering::Relaxed);
        } else {
            stats.url_no_match.fetch_add(1, Ordering::Relaxed);
        }
        false
    }
}


fn check_any_affiliation_string_present(authorships: Option<&Vec<Authorship>>, stats: &Stats) -> bool {
    match authorships {
        Some(auths) if !auths.is_empty() => {
            for author in auths {
                match author.raw_affiliation_strings.as_deref() {
                    Some(raw_strings) => {
                        if !raw_strings.is_empty() {
                            return true;
                        }
                    }
                    None => {
                        stats.affiliation_raw_missing.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            stats.affiliation_all_empty_or_missing.fetch_add(1, Ordering::Relaxed);
            false
        }
        _ => {
            stats.affiliation_author_missing.fetch_add(1, Ordering::Relaxed);
            false
        }
    }
}


fn extract_doi_prefix(record: &OpenAlexRecord) -> Option<DoiPrefix> {
    record.doi.as_ref().and_then(|doi_str| {
        let cleaned_doi_str = doi_str.trim();

        let doi_start_index = cleaned_doi_str.rfind("10.").and_then(|idx| {
            if idx == 0 {
                Some(idx)
            } else {
                match cleaned_doi_str.chars().nth(idx.saturating_sub(1)) {
                    Some('/') | Some(':') | Some(' ') => Some(idx),
                    _ => {
                        Some(idx)
                    }
                }
            }
        });


        doi_start_index.and_then(|start_index| {
            let potential_doi_part = &cleaned_doi_str[start_index..];

            potential_doi_part.split_once('/').and_then(|(pfx, _)| {
                let trimmed_pfx = pfx.trim();
                    if !trimmed_pfx.is_empty() && trimmed_pfx.starts_with("10.") && trimmed_pfx.len() > 3 {
                        Some(DoiPrefix(trimmed_pfx.to_string()))
                } else {
                    warn!("Extracted DOI prefix '{}' from '{}' seems invalid after split (too short or empty), classifying as unknown.", trimmed_pfx, doi_str);
                    None
                }
            })
            .or_else(|| {
                let trimmed_pfx = potential_doi_part.trim();
                    if !trimmed_pfx.is_empty() && trimmed_pfx.starts_with("10.") && trimmed_pfx.len() > 3 {
                    info!("DOI string '{}' appears to contain only the prefix '{}'. Using it.", doi_str, trimmed_pfx);
                    Some(DoiPrefix(trimmed_pfx.to_string()))
                } else {
                    warn!("Could not find '/' after '10.' in DOI string '{}' and remaining part invalid, classifying as unknown.", doi_str);
                    None
                }
                })
        })
        .or_else(|| {
                if !cleaned_doi_str.is_empty() {
                    warn!("Could not find '10.' marker in non-empty DOI string '{}', classifying as unknown.", doi_str);
                }
                None
        })
    })
}

struct PrefixWriterManager {
    base_output_dir: PathBuf,
    max_open: usize,
    writers: Mutex<HashMap<DoiPrefix, BufWriter<GzEncoder<File>>>>,
    lru: Mutex<VecDeque<DoiPrefix>>,
    created_dirs: Mutex<HashSet<PathBuf>>,
    stats: Arc<Stats>,
}

impl PrefixWriterManager {
    fn new(
        base_output_dir: PathBuf,
        max_open: usize,
        stats: Arc<Stats>,
    ) -> Result<Self> {
        Ok(Self {
            base_output_dir,
            max_open: max_open.max(1),
            writers: Mutex::new(HashMap::new()),
            lru: Mutex::new(VecDeque::new()),
            created_dirs: Mutex::new(HashSet::new()),
            stats,
        })
    }

    fn close_writer(prefix: &DoiPrefix, writer: BufWriter<GzEncoder<File>>) -> Option<String> {

        debug!("Attempting to close writer for prefix {}", prefix.0);
        match writer.into_inner() {
            Ok(gz_encoder) => {
                debug!("Successfully flushed BufWriter for prefix {}. Finishing GzEncoder...", prefix.0);
                match gz_encoder.finish() {
                    Ok(_file) => {
                        debug!("Successfully finished GZ stream and closed file for prefix {}", prefix.0);
                        None
                    }
                    Err(e) => {
                        let msg = format!("I/O Error finishing GZ stream for prefix {}: {}", prefix.0, e);
                        error!("{}", msg);
                        Some(msg)
                    }
                }
            }
            Err(into_inner_err) => {
                let msg = format!("Error flushing BufWriter for prefix {} on close: {}", prefix.0, into_inner_err);
                error!("{}", msg);
                Some(msg)
            }
        }
    }


    fn write_line(&self, prefix: &DoiPrefix, line_bytes: &[u8]) -> Result<()> {
        let mut writers_guard = self.writers.lock().unwrap();
        let mut lru_guard = self.lru.lock().unwrap();

        let was_present = writers_guard.contains_key(prefix);

        if !was_present && writers_guard.len() >= self.max_open {
            while writers_guard.len() >= self.max_open {
                if let Some(evict_prefix) = lru_guard.pop_back() {
                    debug!("Evicting prefix writer for {}", evict_prefix.0);
                    if let Some(writer_to_close) = writers_guard.remove(&evict_prefix) {
                        drop(lru_guard);
                        drop(writers_guard);

                        Self::close_writer(&evict_prefix, writer_to_close);

                        writers_guard = self.writers.lock().unwrap();
                        lru_guard = self.lru.lock().unwrap();
                    } else {
                        warn!("LRU contained evicted prefix {} but it wasn't in the writers map. Inconsistency?", evict_prefix.0);
                    }
                } else {
                    error!("LRU queue is empty while trying to evict, but writers map is full ({} writers, max {}). Cannot evict.", writers_guard.len(), self.max_open);
                    break;
                }
            }
        }

        let writer = writers_guard.entry(prefix.clone()).or_insert_with(|| {
            let prefix_dir = self.base_output_dir.join(&prefix.0);
            {
                let mut created_dirs_guard = self.created_dirs.lock().unwrap();
                if !created_dirs_guard.contains(&prefix_dir) {
                    drop(created_dirs_guard);
                    match fs::create_dir_all(&prefix_dir) {
                        Ok(_) => {
                                debug!("Created output directory: {}", prefix_dir.display());
                                self.created_dirs.lock().unwrap().insert(prefix_dir.clone());
                            },
                        Err(e) => {
                                error!("Failed to create output directory {}: {}", prefix_dir.display(), e);
                            }
                    }
                }
            }

            let final_file_path = prefix_dir.join("data.jsonl.gz");
            let file_exists = final_file_path.exists();

            let file = OpenOptions::new().write(true).create(true).append(true).open(&final_file_path)
                .with_context(|| format!("Failed to open/create output file: {}", final_file_path.display()))
                .expect("CRITICAL: Failed to open/create output file");

            debug!("Opened output file{} {}", if file_exists { " (appending)" } else { " (new)" }, final_file_path.display());

            self.stats.prefix_files_opened.fetch_add(1, Ordering::Relaxed);
            self.stats.unique_prefixes_written.lock().unwrap().insert(prefix.clone());

            lru_guard.push_front(prefix.clone());

            let gz_encoder = GzEncoder::new(file, Compression::default());
            BufWriter::new(gz_encoder)
        });

        if was_present {
                if lru_guard.front() != Some(prefix) {
                    if let Some(pos) = lru_guard.iter().position(|id| id == prefix) {
                        let id = lru_guard.remove(pos).unwrap();
                        lru_guard.push_front(id);
                    } else {
                        warn!("Prefix {} writer existed but wasn't in LRU queue? Adding it to front.", prefix.0);
                        lru_guard.push_front(prefix.clone());
                    }
                }
            }


        writer.write_all(line_bytes)?;
        if !line_bytes.ends_with(b"\n") {
            writer.write_all(b"\n")?;
        }

        Ok(())
    }

    fn flush_all(&self) -> Result<()> {
        info!("Flushing all open prefix writers...");
        let mut writers_guard = self.writers.lock().unwrap();
        let mut lru_guard = self.lru.lock().unwrap();
        let mut errors: Vec<String> = Vec::new();

        for (prefix, writer) in writers_guard.drain() {
                debug!("Flushing and closing writer for prefix: {}", prefix.0);
                if let Some(err_msg) = Self::close_writer(&prefix, writer) {
                    errors.push(err_msg);
                }
        }
        lru_guard.clear();

        if errors.is_empty() {
            info!("All prefix writers flushed and closed successfully.");
            Ok(())
        } else {
            Err(anyhow::anyhow!("Errors occurred during final prefix writer flush/close:\n - {}", errors.join("\n - ")))
        }
    }
}

impl Drop for PrefixWriterManager {
    fn drop(&mut self) {
        info!("PrefixWriterManager dropping. Attempting final flush...");
        if let Ok(mut writers_guard) = self.writers.try_lock() {
            if let Ok(mut lru_guard) = self.lru.try_lock() {
                let mut errors = Vec::new();
                for (prefix, writer) in writers_guard.drain() {
                        if let Some(err_msg) = Self::close_writer(&prefix, writer) {
                            errors.push(err_msg);
                        }
                }
                lru_guard.clear();
                if !errors.is_empty() {
                    error!("Errors occurred during final flush in PrefixWriterManager drop:\n - {}", errors.join("\n - "));
                } else {
                    info!("Final flush in PrefixWriterManager drop completed.");
                }
            } else {
                error!("Could not lock LRU queue during PrefixWriterManager drop flush. Some files might not be closed cleanly.");
            }
        } else {
            error!("Could not lock writers map during PrefixWriterManager drop flush. Some files might not be closed cleanly.");
        }
    }
}


mod memory_usage {
    #[cfg(target_os = "linux")] pub mod linux { use std::fs::read_to_string; use super::MemoryStats; pub fn get_memory_usage() -> Option<MemoryStats> { let pid=std::process::id(); let status_file=format!("/proc/{}/status",pid); let content=read_to_string(status_file).ok()?; let mut vm_rss_kb=None; let mut vm_size_kb=None; for line in content.lines() { if line.starts_with("VmRSS:") { vm_rss_kb=line.split_whitespace().nth(1).and_then(|s|s.parse::<f64>().ok()); } else if line.starts_with("VmSize:") { vm_size_kb=line.split_whitespace().nth(1).and_then(|s|s.parse::<f64>().ok()); } if vm_rss_kb.is_some()&&vm_size_kb.is_some() { break; }} let rss_mb=vm_rss_kb.map(|kb| kb / 1024.0); let vm_size_mb=vm_size_kb.map(|kb| kb / 1024.0); let mut percent=None; if let Ok(meminfo)=read_to_string("/proc/meminfo") { if let Some(mem_total_kb)=meminfo.lines().find(|line|line.starts_with("MemTotal:")).and_then(|line|line.split_whitespace().nth(1)).and_then(|s|s.parse::<f64>().ok()) { if mem_total_kb > 0.0 { if let Some(rss) = vm_rss_kb { percent=Some((rss / mem_total_kb)* 100.0); }}}} Some(MemoryStats { rss_mb: rss_mb.unwrap_or(0.0), vm_size_mb: vm_size_mb.unwrap_or(0.0), percent })}}
    #[cfg(target_os = "macos")] pub mod macos { use std::process::Command; use super::MemoryStats; pub fn get_memory_usage() -> Option<MemoryStats> { let pid=std::process::id(); let ps_output_rss=Command::new("ps").args(&["-o","rss=","-p",&pid.to_string()]).output().ok()?; let rss_kb=String::from_utf8_lossy(&ps_output_rss.stdout).trim().parse::<f64>().ok()?; let ps_output_vsz=Command::new("ps").args(&["-o","vsz=","-p",&pid.to_string()]).output().ok()?; let vsz_kb=String::from_utf8_lossy(&ps_output_vsz.stdout).trim().parse::<f64>().ok()?; let rss_mb=rss_kb/1024.0; let vm_size_mb=vsz_kb/1024.0; let mut percent=None; if let Ok(hw_mem_output)=Command::new("sysctl").args(&["-n","hw.memsize"]).output() { if let Ok(total_bytes_str)=String::from_utf8(hw_mem_output.stdout) { if let Ok(total_bytes)=total_bytes_str.trim().parse::<f64>() { let total_kb=total_bytes / 1024.0; if total_kb > 0.0 { percent=Some((rss_kb/total_kb)* 100.0); }}}} Some(MemoryStats { rss_mb, vm_size_mb, percent })}}
    #[cfg(target_os = "windows")] pub mod windows { use std::process::Command; use super::MemoryStats; pub fn get_memory_usage() -> Option<MemoryStats> { let pid=std::process::id(); let wmic_output=Command::new("wmic").args(&["process","where",&format!("ProcessId={}", pid),"get","WorkingSetSize,","PageFileUsage","/value",]).output().ok()?; let output_str=String::from_utf8_lossy(&wmic_output.stdout); let mut rss_bytes:Option<f64>=None; let mut vm_kb:Option<f64>=None; for line in output_str.lines() { let parts:Vec<&str>=line.split('=').collect(); if parts.len()==2 { let key=parts[0].trim(); let value=parts[1].trim(); match key { "PageFileUsage"=>vm_kb=value.parse::<f64>().ok(), "WorkingSetSize"=>rss_bytes=value.parse::<f64>().ok(), _=> {} }}} let rss_mb=rss_bytes.map(|b| b / (1024.0*1024.0)); let vm_size_mb=vm_kb.map(|kb| kb / 1024.0); let mut percent=None; if let Ok(mem_output)=Command::new("wmic").args(&["ComputerSystem","get","TotalPhysicalMemory","/value"]).output() { let mem_str=String::from_utf8_lossy(&mem_output.stdout); if let Some(total_bytes_str)=mem_str.lines().find(|line|line.starts_with("TotalPhysicalMemory=")).and_then(|line|line.split('=').nth(1)) { if let Ok(total_bytes)=total_bytes_str.trim().parse::<f64>() { if total_bytes > 0.0 { if let Some(rss) = rss_bytes { percent = Some((rss / total_bytes) * 100.0); }}}}} Some(MemoryStats { rss_mb: rss_mb.unwrap_or(0.0), vm_size_mb: vm_size_mb.unwrap_or(0.0), percent })}}
    #[derive(Debug)] pub struct MemoryStats { pub rss_mb: f64, pub vm_size_mb: f64, pub percent: Option<f64>, }
    #[cfg(target_os = "linux")] use self::linux::get_memory_usage; #[cfg(target_os = "macos")] use self::macos::get_memory_usage; #[cfg(target_os = "windows")] use self::windows::get_memory_usage;
    #[cfg(not(any(target_os="linux",target_os="macos",target_os="windows")))] pub fn get_memory_usage()->Option<MemoryStats> { None }
    pub fn log_memory_usage(note: &str) { use log::info; if let Some(stats)=get_memory_usage() { let percent_str=stats.percent.map_or_else(||"N/A".to_string(), |p|format!("{:.1}%",p)); let vm_str=if stats.vm_size_mb > 0.0 { format!("{:.1} MB virtual/commit",stats.vm_size_mb) } else { "N/A virtual".to_string() }; info!("Memory usage ({}): {:.1} MB physical (RSS), {}, {} of system memory", note, stats.rss_mb, vm_str, percent_str ); } else { info!("Memory usage tracking not available or failed on this platform ({})", std::env::consts::OS); } }
}

fn format_elapsed(elapsed: Duration) -> String {
    let total_secs = elapsed.as_secs();
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;
    let millis = elapsed.subsec_millis();

    if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, seconds)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, seconds)
    } else {
        format!("{}.{:03}s", seconds, millis)
    }
}


fn main() -> Result<()> {
    let main_start_time = Instant::now();
    let cli = Cli::parse();

    let log_level = match cli.log_level.to_uppercase().as_str() {
        "DEBUG"=>LevelFilter::Debug,
        "INFO"=>LevelFilter::Info,
        "WARN"|"WARNING"=>LevelFilter::Warn,
        "ERROR"=>LevelFilter::Error,
        _ => { eprintln!("Invalid log level '{}', defaulting to INFO.", cli.log_level); LevelFilter::Info }
    };
    SimpleLogger::new()
        .with_level(log_level)
        .with_timestamp_format(format_description!("[year]-[month]-[day] [hour]:[minute]:[second]"))
        .init()?;

    info!("Starting OpenAlex URL/Affiliation Filter v2.1.0");
    memory_usage::log_memory_usage("initial");

    let output_dir = PathBuf::from(&cli.output_dir);
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("Failed to create output directory: {}", output_dir.display()))?;

    let num_threads = if cli.threads == 0 {
        let cores = num_cpus::get();
        info!("Auto-detected {} CPU cores. Using {} threads.", cores, cores);
        cores
    } else {
        info!("Using specified {} threads.", cli.threads);
        cli.threads
    };
    if let Err(e) = rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global() {
        error!("Failed to build global thread pool with {} threads: {}. Proceeding with default.", num_threads, e);
    }

    let base_urls = read_base_urls_from_csv(&cli.base_urls_csv)?;
    if base_urls.is_empty() {
        error!("No valid base URLs were loaded from {}. Cannot proceed.", cli.base_urls_csv);
        return Err(anyhow!("No base URLs loaded."));
    }
    info!("Loaded {} base URL(s) for filtering from {}", base_urls.len(), cli.base_urls_csv);
    let base_urls_arc = Arc::new(base_urls);

    info!("Searching for input files in: {}", cli.input_dir);
    let files = find_gz_files_excluding_csv_gz(&cli.input_dir)?;

    if files.is_empty() {
        warn!("No .jsonl.gz files found recursively in the specified directory. Exiting.");
        return Ok(());
    }
    info!("Found {} input .jsonl.gz files.", files.len());

    info!("Output directory: {}", output_dir.display());
    info!("Max open prefix files: {}", cli.max_open_prefix_files);
    info!("Statistics logging interval: {} seconds.", cli.stats_interval);

    let stats = Arc::new(Stats::new());

    let stats_thread_running = Arc::new(Mutex::new(true));
    let stats_interval_duration = Duration::from_secs(cli.stats_interval);
    let stats_clone_for_thread = Arc::clone(&stats);
    let stats_thread_running_clone = Arc::clone(&stats_thread_running);
    let stats_thread = thread::spawn(move || {
        info!("Stats logging thread started.");
        let mut last_log_time = Instant::now();
        loop {
            if let Ok(guard) = stats_thread_running_clone.try_lock() {
                if !*guard { info!("Stats thread received stop signal."); break; }
            }

            thread::sleep(Duration::from_millis(500));

            if last_log_time.elapsed() >= stats_interval_duration {
                memory_usage::log_memory_usage("periodic check");
                stats_clone_for_thread.log_current_stats("Processing");
                last_log_time = Instant::now();
            }
        }
        info!("Stats logging thread finished.");
    });

    info!("--- Starting Processing: Filtering records and writing to prefix files ---");
    let processing_start_time = Instant::now();

    let prefix_writer_manager = Arc::new(
        PrefixWriterManager::new(
            output_dir.clone(),
            cli.max_open_prefix_files,
            Arc::clone(&stats),
        )?
    );

    let progress_bar = ProgressBar::new(files.len() as u64);
    progress_bar.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta} @ {per_sec}) {msg}")
            .expect("Failed to create progress bar template")
            .progress_chars("=> "),
    );
    progress_bar.set_message("Processing: Starting...");

    let processing_results: Vec<Result<(), String>> = files
        .par_iter()
        .map(|filepath| {
            let pb_clone = progress_bar.clone();
            let writer_manager_clone = Arc::clone(&prefix_writer_manager);
            let stats_clone = Arc::clone(&stats);
            let base_urls_clone = Arc::clone(&base_urls_arc);

            let file = match File::open(filepath) {
                Ok(f) => f,
                Err(e) => return Err(format!("Failed to open {}: {}", filepath.display(), e)),
            };
            let decoder = GzDecoder::new(file);
            let mut reader = BufReader::new(decoder);
            let mut byte_buffer = Vec::with_capacity(8192);

            let mut file_errors: Vec<String> = Vec::new();

            loop {
                byte_buffer.clear();
                match reader.read_until(b'\n', &mut byte_buffer) {
                    Ok(0) => break,
                    Ok(bytes_read) => {
                        if bytes_read == 0 { break; }
                        stats_clone.lines_read.fetch_add(1, Ordering::Relaxed);

                        if byte_buffer.iter().all(|&b| b.is_ascii_whitespace()) {
                            continue;
                        }

                        match serde_json::from_slice::<OpenAlexRecord>(&byte_buffer) {
                            Ok(record) => {
                                let passes_url_check = check_location_url_match_any(
                                    record.locations.as_ref(),
                                    &base_urls_clone,
                                    &stats_clone,
                                );
                                if !passes_url_check { continue; }

                                let passes_affiliation_check = check_any_affiliation_string_present(
                                    record.authorships.as_ref(),
                                    &stats_clone,
                                );
                                if !passes_affiliation_check { continue; }

                                stats_clone.lines_passed_filter.fetch_add(1, Ordering::Relaxed);

                                let prefix = match extract_doi_prefix(&record) {
                                    Some(p) => p,
                                    None => {
                                        stats_clone.doi_missing.fetch_add(1, Ordering::Relaxed);
                                        DoiPrefix("_unknown_".to_string())
                                    }
                                };

                                match writer_manager_clone.write_line(&prefix, &byte_buffer) {
                                    Ok(_) => { }
                                    Err(e) => {
                                        let msg = format!("Failed write for prefix {} from {}: {}", prefix.0, filepath.display(), e);
                                        error!("{}", msg);
                                        file_errors.push(msg);
                                    }
                                }

                            }
                            Err(e) => {
                                stats_clone.json_parse_errors.fetch_add(1, Ordering::Relaxed);
                                if stats_clone.json_parse_errors.load(Ordering::Relaxed) % 5000 == 1 {
                                    let snippet = String::from_utf8_lossy(&byte_buffer).chars().take(150).collect::<String>();
                                    warn!("JSON parse error in {}: {} (Line starts: '{}...')", filepath.display(), e, snippet);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let msg = format!("Read error in {}: {}", filepath.display(), e);
                        error!("{}", msg);
                        file_errors.push(msg);
                        break;
                    }
                }
            }

            pb_clone.inc(1);
            let file_name_msg = filepath.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| filepath.display().to_string());
            pb_clone.set_message(format!("Processed: {}", file_name_msg));

            if file_errors.is_empty() {
                Ok(())
            } else {
                Err(format!("Errors processing {}:\n - {}", filepath.display(), file_errors.join("\n - ")))
            }
        })
        .collect();

    info!("Flushing remaining writers...");
    let flush_result = prefix_writer_manager.flush_all();
    progress_bar.finish_with_message("Processing: Finished filtering and writing.");

    let processing_duration = processing_start_time.elapsed();
    stats.log_current_stats("Processing Complete");

    let final_errors = processing_results.iter().filter(|r| r.is_err()).count()
                            + if flush_result.is_err() { 1 } else { 0 };

    info!("--- Processing Summary ---");
    info!("Duration: {}", format_elapsed(processing_duration));
    info!("Input files processed: {} ({} reported file/write/flush errors)", files.len(), final_errors);
    if final_errors > 0 {
        warn!("{} files/operations encountered errors during processing.", final_errors);
        for result in processing_results.iter().filter_map(|r| r.as_ref().err()) {
            error!("  - {}", result);
        }
        if let Err(e) = flush_result {
            error!("  - Final Flush Error: {}", e);
        }
        warn!("Processing finished with errors. Output might be incomplete or contain partial data.");
    } else {
        info!("Processing finished successfully.");
    }

    info!("Signaling stats thread to stop...");
    if let Ok(mut running_guard) = stats_thread_running.lock() { *running_guard = false; }
    else { error!("Failed to lock stats thread running flag to signal stop."); }
    info!("Waiting for stats thread to finish...");
    if let Err(e) = stats_thread.join() { error!("Error joining stats thread: {:?}", e); }
    else { info!("Stats thread joined successfully."); }

    info!("-------------------- FINAL SUMMARY --------------------");
    let total_runtime = main_start_time.elapsed();
    info!("Total execution time: {}", format_elapsed(total_runtime));
    info!("Input files found: {}", files.len());
    info!("Base URL filter applied: Matched landing_page_url against {} URL(s) from {}", base_urls_arc.len(), cli.base_urls_csv);
    info!("Affiliation filter applied: At least one authorship must have non-empty raw_affiliation_strings");
    info!("Processing errors (file/write/flush): {}", final_errors);

    stats.log_current_stats("Final");
    memory_usage::log_memory_usage("final");
    info!("Filtering and organization process finished.");
    info!("-------------------------------------------------------");

    if final_errors > 0 {
        Err(anyhow!("Processing finished with errors."))
    } else {
        Ok(())
    }
}