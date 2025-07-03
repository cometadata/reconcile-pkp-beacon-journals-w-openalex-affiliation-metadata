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
#[command(name = "Get OpenAlex works with PKP Beacon base URL")]
#[command(about = "Filters OpenAlex JSONL.gz files based on locations[].landing_page_url matching ANY URL in a CSV and *at least one non-empty* authorships[].raw_affiliation_strings, organizing by DOI prefix.")]
#[command(version = "1.1.0")]
struct Cli {
    #[arg(short, long, help = "Directory containing input JSONL.gz files", required = true)]
    input_dir: String,

    #[arg(short, long, help = "Base directory for organized output structure (prefix/data.jsonl.gz)", required = true)]
    output_dir: String,

    #[arg(long, short = 'b', help = "Path to CSV file containing base URLs (header 'base_url')", required = true)]
    base_urls_csv: String,

    #[arg(long, default_value = "64", help = "Max open final prefix output files")]
    max_open_prefix_files: usize,

    #[arg(short, long, default_value = "INFO", help = "Logging level (DEBUG, INFO, WARN, ERROR)")]
    log_level: String,

    #[arg(short, long, default_value = "0", help = "Number of threads to use (0 for auto)")]
    threads: usize,

    #[arg(short, long, default_value = "60", help = "Interval in seconds to log statistics")]
    stats_interval: u64,

    #[arg(long, default_value = "6", help = "GZIP compression level for output files (0-9)")]
    compression_level: u32,

    #[arg(long, default_value = "8", help = "Output file writer buffer size in KB")]
    writer_buffer_kb: usize,
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
            path.file_name()
                .and_then(|name| name.to_str())
                .map_or(false, |name_str| !name_str.ends_with(".csv.gz"))
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

    for location in locations_vec {
        if let Some(landing_url_str) = location.landing_page_url.as_deref() {
            if landing_url_str.is_empty() {
                continue;
            }
            found_any_landing_page = true;
            match Url::parse(landing_url_str) {
                Ok(landing_url) => {
                    let record_scheme = landing_url.scheme();
                    let record_host = landing_url.host_str();
                    let record_port = landing_url.port_or_known_default();

                    if base_urls.iter().any(|base_url| {
                        record_scheme == base_url.scheme() &&
                        record_host == base_url.host_str() &&
                        record_port == base_url.port_or_known_default()
                    }) {
                        return true;
                    }
                }
                Err(_e) => {
                    stats.url_landing_page_parse_errors.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    if !found_any_landing_page {
        stats.url_landing_page_missing.fetch_add(1, Ordering::Relaxed);
    } else {
        stats.url_no_match.fetch_add(1, Ordering::Relaxed);
    }
    false
}


fn check_any_affiliation_string_present(authorships: Option<&Vec<Authorship>>, stats: &Stats) -> bool {
    match authorships {
        Some(auths) if !auths.is_empty() => {
            for author in auths {
                match author.raw_affiliation_strings.as_deref() {
                    Some(raw_strings) => {
                        if raw_strings.iter().any(|s| !s.trim().is_empty()) {
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
        if cleaned_doi_str.is_empty() { return None; }

        let doi_start_index = cleaned_doi_str.rfind("10.").filter(|&idx| {
            if cleaned_doi_str.len() > idx + 3 {
                cleaned_doi_str.chars().nth(idx + 3).map_or(false, |c| c.is_digit(10) || c == '/')
            } else { true }
        });

        doi_start_index.and_then(|start_index| {
            let potential_doi_part = &cleaned_doi_str[start_index..];
            potential_doi_part.split_once('/').map(|(pfx, _)| pfx.trim())
                .or_else(|| {
                    if potential_doi_part.contains('/') { None }
                    else { Some(potential_doi_part.trim()) }
                })
                .filter(|pfx_str| !pfx_str.is_empty() && pfx_str.starts_with("10.") && pfx_str.len() > 3)
                .map(|valid_pfx_str| DoiPrefix(valid_pfx_str.to_string()))
        })
        .or_else(|| {
            warn!("Could not extract a valid DOI prefix from DOI string: '{}'. Classifying as unknown.", doi_str);
            None
        })
    })
}

type WriterType = BufWriter<GzEncoder<File>>;

struct PrefixWriterManager {
    base_output_dir: PathBuf,
    max_open: usize,
    writers: Mutex<HashMap<DoiPrefix, WriterType>>,
    lru: Mutex<VecDeque<DoiPrefix>>,
    created_dirs: Mutex<HashSet<PathBuf>>,
    stats: Arc<Stats>,
    compression_level: Compression,
    writer_buffer_kb: usize,
}

impl PrefixWriterManager {
    fn new(
        base_output_dir: PathBuf,
        max_open: usize,
        stats: Arc<Stats>,
        compression_level: u32,
        writer_buffer_kb: usize,
    ) -> Result<Self> {
        Ok(Self {
            base_output_dir,
            max_open: max_open.max(1),
            writers: Mutex::new(HashMap::new()),
            lru: Mutex::new(VecDeque::new()),
            created_dirs: Mutex::new(HashSet::new()),
            stats,
            compression_level: Compression::new(compression_level.clamp(0, 9)),
            writer_buffer_kb,
        })
    }

    fn close_writer(prefix: &DoiPrefix, writer: WriterType) -> Result<(), String> {
        debug!("Attempting to close writer for prefix {}", prefix.0);
        match writer.into_inner() {
            Ok(gz_encoder) => {
                debug!("Successfully flushed BufWriter for {}. Finishing GzEncoder...", prefix.0);
                match gz_encoder.finish() {
                    Ok(_file) => {
                        debug!("Successfully finished GZ stream and closed file for prefix {}", prefix.0);
                        Ok(())
                    }
                    Err(e) => Err(format!("I/O Error finishing GZ stream for prefix {}: {}", prefix.0, e)),
                }
            }
            Err(into_inner_err) => Err(format!("Error flushing BufWriter for prefix {} on close: {}", prefix.0, into_inner_err.error())),
        }
    }
    
    fn create_file_writer_for_prefix(&self, prefix: &DoiPrefix) -> Result<WriterType> {
        let prefix_dir = self.base_output_dir.join(&prefix.0);

        let mut created_dirs_guard = self.created_dirs.lock().unwrap();
        if !created_dirs_guard.contains(&prefix_dir) {
            drop(created_dirs_guard);
            fs::create_dir_all(&prefix_dir)
                .with_context(|| format!("Failed to create output directory {}", prefix_dir.display()))?;
            self.created_dirs.lock().unwrap().insert(prefix_dir.clone());
        }

        let final_file_path = prefix_dir.join("data.jsonl.gz");
        let file_exists = final_file_path.exists();

        let file = OpenOptions::new().write(true).create(true).append(true).open(&final_file_path)
            .with_context(|| format!("Failed to open/create output file: {}", final_file_path.display()))?;

        debug!("IO: Opened output file{} for prefix {} (new writer instance): {}", if file_exists { " (appending)" } else { " (new)" }, prefix.0, final_file_path.display());

        let gz_encoder = GzEncoder::new(file, self.compression_level);
        Ok(BufWriter::with_capacity(self.writer_buffer_kb * 1024, gz_encoder))
    }

    fn touch_lru(lru_queue: &mut VecDeque<DoiPrefix>, prefix: &DoiPrefix, max_size: usize) {
        if lru_queue.front() != Some(prefix) {
            if let Some(pos) = lru_queue.iter().position(|id| id == prefix) {
                let id = lru_queue.remove(pos).expect("Prefix was in LRU, so remove should succeed");
                lru_queue.push_front(id);
            } else {
                lru_queue.push_front(prefix.clone());
                while lru_queue.len() > max_size {
                    lru_queue.pop_back();
                }
            }
        }
    }


    fn write_line(&self, prefix: &DoiPrefix, line_bytes: &[u8]) -> Result<()> {
        {
            let mut writers_guard = self.writers.lock().unwrap();
            if writers_guard.contains_key(prefix) {
                let writer = writers_guard.get_mut(prefix).unwrap();
                let mut lru_guard = self.lru.lock().unwrap();
                Self::touch_lru(&mut lru_guard, prefix, self.max_open);
                drop(lru_guard);

                writer.write_all(line_bytes)?;
                if !line_bytes.ends_with(b"\n") {
                    writer.write_all(b"\n")?;
                }
                return Ok(());
            }
        }

        let new_physical_writer = match self.create_file_writer_for_prefix(prefix) {
            Ok(writer) => writer,
            Err(e) => {
                error!("Failed to create physical file writer for prefix {}: {}. Line will be dropped for this prefix.", prefix.0, e);
                return Err(e);
            }
        };

        let mut writers_guard = self.writers.lock().unwrap();
        let mut lru_guard = self.lru.lock().unwrap();

        if writers_guard.contains_key(prefix) {
            debug!("Race detected: Writer for {} created by another thread. Closing our redundant new file writer.", prefix.0);
            drop(lru_guard);
            drop(writers_guard);

            if let Err(e) = Self::close_writer(prefix, new_physical_writer) {
                 error!("Error closing redundant new file writer for {}: {}", prefix.0, e);
            }
            debug!("Re-calling write_line for prefix {} after race.", prefix.0);
            return self.write_line(prefix, line_bytes);
        }

        while writers_guard.len() >= self.max_open {
            if let Some(evict_prefix) = lru_guard.pop_back() {
                debug!("Cache full. Evicting prefix writer for {} (to make space for {})", evict_prefix.0, prefix.0);
                if let Some(writer_to_close) = writers_guard.remove(&evict_prefix) {
                    if let Err(e) = Self::close_writer(&evict_prefix, writer_to_close) {
                        error!("Failed to close evicted writer for {}: {}", evict_prefix.0, e);
                    }
                } else {
                    warn!("LRU contained {} for eviction, but it wasn't in writers map. (Race with another eviction?)", evict_prefix.0);
                }
            } else {
                error!("LRU queue empty during commit-time eviction, but writers map full ({}). Cannot make space for {}.", writers_guard.len(), prefix.0);
                return Err(anyhow!("Cannot evict writer to make space for new prefix {}, LRU empty but cache full.", prefix.0));
            }
        }

        self.stats.prefix_files_opened.fetch_add(1, Ordering::Relaxed);
        if self.stats.unique_prefixes_written.lock().unwrap().insert(prefix.clone()) {
             debug!("Tracking new unique prefix (commit phase): {}", prefix.0);
        }
        
        writers_guard.insert(prefix.clone(), new_physical_writer);
        lru_guard.push_front(prefix.clone());

        let writer = writers_guard.get_mut(prefix).expect("Writer just inserted should be present");
        
        writer.write_all(line_bytes)?;
        if !line_bytes.ends_with(b"\n") {
            writer.write_all(b"\n")?;
        }

        Ok(())
    }

    fn flush_all(&self) -> Result<()> {
        info!("Flushing all open prefix writers explicitly...");
        let mut writers_guard = self.writers.lock().unwrap();
        let mut lru_guard = self.lru.lock().unwrap();
        let mut error_messages: Vec<String> = Vec::new();

        for (prefix, writer) in writers_guard.drain() {
            debug!("Flushing and closing writer for prefix: {}", prefix.0);
            if let Err(err_msg) = Self::close_writer(&prefix, writer) {
                error!("Error closing writer for prefix {}: {}", prefix.0, err_msg);
                error_messages.push(format!("Prefix {}: {}", prefix.0, err_msg));
            }
        }
        lru_guard.clear();

        if error_messages.is_empty() {
            info!("All prefix writers flushed and closed successfully via flush_all().");
            Ok(())
        } else {
            Err(anyhow!("Errors occurred during final prefix writer flush/close:\n - {}", error_messages.join("\n - ")))
        }
    }
}

impl Drop for PrefixWriterManager {
    fn drop(&mut self) {
        info!("PrefixWriterManager dropping. Checking for any remaining open writers...");
        let mut writers_guard = self.writers.lock().unwrap_or_else(|p| {
            error!("Writers Mutex was poisoned before drop. Attempting to recover.");
            p.into_inner()
        });

        if !writers_guard.is_empty() {
            warn!("PrefixWriterManager drop: {} writers still open! This indicates flush_all() was not called or failed. Attempting emergency close.", writers_guard.len());
            let mut lru_guard = self.lru.lock().unwrap_or_else(|p| p.into_inner());
            let mut errors_in_drop: Vec<String> = Vec::new();

            for (prefix, writer) in writers_guard.drain() {
                error!("Drop: Force closing writer for prefix {}", prefix.0);
                if let Err(e) = Self::close_writer(&prefix, writer) {
                    errors_in_drop.push(format!("Drop-Close Error for {}: {}", prefix.0, e));
                }
            }
            lru_guard.clear();
            if !errors_in_drop.is_empty() {
                error!("Errors occurred during emergency writer close in PrefixWriterManager drop:\n - {}", errors_in_drop.join("\n - "));
            } else {
                info!("Emergency close in PrefixWriterManager drop completed for remaining writers.");
            }
        } else {
            info!("PrefixWriterManager drop: No writers were open, flush_all() likely completed successfully.");
        }
    }
}


mod memory_usage {
    #[cfg(target_os = "linux")] pub mod linux { use std::fs::read_to_string; use super::MemoryStats; pub fn get_memory_usage() -> Option<MemoryStats> { let pid=std::process::id(); let status_file=format!("/proc/{}/status",pid); let content=read_to_string(status_file).ok()?; let mut vm_rss_kb=None; let mut vm_size_kb=None; for line in content.lines() { if line.starts_with("VmRSS:") { vm_rss_kb=line.split_whitespace().nth(1).and_then(|s|s.parse::<f64>().ok()); } else if line.starts_with("VmSize:") { vm_size_kb=line.split_whitespace().nth(1).and_then(|s|s.parse::<f64>().ok()); } if vm_rss_kb.is_some()&&vm_size_kb.is_some() { break; }} let rss_mb=vm_rss_kb.map(|kb| kb / 1024.0); let vm_size_mb=vm_size_kb.map(|kb| kb / 1024.0); let mut percent=None; if let Ok(meminfo)=read_to_string("/proc/meminfo") { if let Some(mem_total_kb)=meminfo.lines().find(|line|line.starts_with("MemTotal:")).and_then(|line|line.split_whitespace().nth(1)).and_then(|s|s.parse::<f64>().ok()) { if mem_total_kb > 0.0 { if let Some(rss) = vm_rss_kb { percent=Some((rss / mem_total_kb)* 100.0); }}}} Some(MemoryStats { rss_mb: rss_mb.unwrap_or(0.0), vm_size_mb: vm_size_mb.unwrap_or(0.0), percent })}}
    #[cfg(target_os = "macos")] pub mod macos { use std::process::Command; use super::MemoryStats; pub fn get_memory_usage() -> Option<MemoryStats> { let pid=std::process::id(); let ps_output_rss=Command::new("ps").args(&["-o","rss=","-p",&pid.to_string()]).output().ok()?; let rss_kb=String::from_utf8_lossy(&ps_output_rss.stdout).trim().parse::<f64>().ok()?; let ps_output_vsz=Command::new("ps").args(&["-o","vsz=","-p",&pid.to_string()]).output().ok()?; let vsz_kb=String::from_utf8_lossy(&ps_output_vsz.stdout).trim().parse::<f64>().ok()?; let rss_mb=rss_kb/1024.0; let vm_size_mb=vsz_kb/1024.0; let mut percent=None; if let Ok(hw_mem_output)=Command::new("sysctl").args(&["-n","hw.memsize"]).output() { if let Ok(total_bytes_str)=String::from_utf8(hw_mem_output.stdout) { if let Ok(total_bytes)=total_bytes_str.trim().parse::<f64>() { let total_kb=total_bytes / 1024.0; if total_kb > 0.0 { percent=Some((rss_kb/total_kb)* 100.0); }}}} Some(MemoryStats { rss_mb, vm_size_mb, percent })}}
    #[cfg(target_os = "windows")] pub mod windows { use std::process::Command as WinCommand; use super::MemoryStats; pub fn get_memory_usage() -> Option<MemoryStats> { let pid=std::process::id(); let wmic_output=WinCommand::new("wmic").args(&["process","where",&format!("ProcessId={}", pid),"get","WorkingSetSize,","PageFileUsage","/value",]).output().ok()?; let output_str=String::from_utf8_lossy(&wmic_output.stdout); let mut rss_bytes:Option<f64>=None; let mut vm_kb:Option<f64>=None; for line in output_str.lines() { let parts:Vec<&str>=line.split('=').collect(); if parts.len()==2 { let key=parts[0].trim(); let value=parts[1].trim(); match key { "PageFileUsage"=>vm_kb=value.parse::<f64>().ok(), "WorkingSetSize"=>rss_bytes=value.parse::<f64>().ok(), _=> {} }}} let rss_mb=rss_bytes.map(|b| b / (1024.0*1024.0)); let vm_size_mb=vm_kb.map(|kb| kb / 1024.0); let mut percent=None; if let Ok(mem_output)=WinCommand::new("wmic").args(&["ComputerSystem","get","TotalPhysicalMemory","/value"]).output() { let mem_str=String::from_utf8_lossy(&mem_output.stdout); if let Some(total_bytes_str)=mem_str.lines().find(|line|line.starts_with("TotalPhysicalMemory=")).and_then(|line|line.split('=').nth(1)) { if let Ok(total_bytes)=total_bytes_str.trim().parse::<f64>() { if total_bytes > 0.0 { if let Some(rss) = rss_bytes { percent = Some((rss / total_bytes) * 100.0); }}}}} Some(MemoryStats { rss_mb: rss_mb.unwrap_or(0.0), vm_size_mb: vm_size_mb.unwrap_or(0.0), percent })}}
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
        .with_timestamp_format(format_description!("[year]-[month]-[day] [hour]:[minute]:[second].[subsecond digits:3]"))
        .init()?;

    info!("Starting OpenAlex URL/Affiliation Filter v{}", env!("CARGO_PKG_VERSION"));
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
        error!("Failed to build global thread pool with {} threads: {}. Proceeding with Rayon's default.", num_threads, e);
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
    info!("Output GZIP compression level: {}", cli.compression_level);
    info!("Output writer buffer size: {} KB", cli.writer_buffer_kb);
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
            cli.compression_level,
            cli.writer_buffer_kb,
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
                Err(e) => {
                    let err_msg = format!("Failed to open {}: {}", filepath.display(), e);
                    error!("{}", err_msg);
                    return Err(err_msg);
                }
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
                                    Ok(_) => {}
                                    Err(e) => {
                                        let msg = format!("Failed write for prefix {} (from file {}): {}", prefix.0, filepath.display(), e);
                                        file_errors.push(msg); 
                                    }
                                }
                            }
                            Err(e) => {
                                stats_clone.json_parse_errors.fetch_add(1, Ordering::Relaxed);
                                if stats_clone.json_parse_errors.load(Ordering::Relaxed) % 10000 == 1 {
                                    let snippet_len = byte_buffer.len().min(150);
                                    let snippet = String::from_utf8_lossy(&byte_buffer[..snippet_len]);
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

    progress_bar.finish_with_message("Processing: All files passed to worker threads.");

    info!("Explicitly flushing all writers...");
    let flush_result = prefix_writer_manager.flush_all();

    let processing_duration = processing_start_time.elapsed();
    stats.log_current_stats("Processing Complete");

    let mut final_errors_count = 0;
    let mut error_report_messages: Vec<String> = Vec::new();

    for (i, res) in processing_results.into_iter().enumerate() {
        if let Err(e_str) = res {
            final_errors_count += 1;
            error_report_messages.push(format!("File Processing Error for '{}': {}", files.get(i).map_or_else(|| "Unknown File".to_string(), |p| p.display().to_string()), e_str));
        }
    }

    if let Err(e) = flush_result {
        final_errors_count += 1;
        error_report_messages.push(format!("Final Writer Flush Error: {}", e));
    }

    info!("--- Processing Summary ---");
    info!("Duration: {}", format_elapsed(processing_duration));
    info!("Input files scheduled for processing: {}", files.len());
    info!("Operations with reported errors (file processing or final flush): {}", final_errors_count);

    if final_errors_count > 0 {
        warn!("{} operations encountered errors during processing or final flush.", final_errors_count);
        for err_msg in error_report_messages {
            error!("  Summary Error: {}", err_msg);
        }
        warn!("Processing finished with errors. Output might be incomplete or contain corrupted GZ files if flush failed.");
    } else {
        info!("Processing and final writer flush finished successfully.");
    }

    info!("Signaling stats thread to stop...");
    if let Ok(mut running_guard) = stats_thread_running.lock() { *running_guard = false; }
    else { error!("Failed to lock stats thread running flag to signal stop."); }
    
    info!("Waiting for stats thread to finish (max 5s)...");
    if let Err(_e) = stats_thread.join().map_err(|_| "Stats thread panicked") {
         error!("Error joining stats thread or thread panicked.");
    } else {
         info!("Stats thread joined successfully.");
    }
    
    drop(prefix_writer_manager);

    info!("-------------------- FINAL SUMMARY --------------------");
    let total_runtime = main_start_time.elapsed();
    info!("Total execution time: {}", format_elapsed(total_runtime));
    info!("Input files found: {}", files.len());
    info!("Base URL filter applied: Matched landing_page_url against {} URL(s) from {}", base_urls_arc.len(), cli.base_urls_csv);
    info!("Affiliation filter applied: At least one authorship must have non-empty raw_affiliation_strings");
    info!("Total operations with errors (file/write/flush): {}", final_errors_count);

    stats.log_current_stats("Final");
    memory_usage::log_memory_usage("final");
    info!("Filtering and organization process finished.");
    info!("-------------------------------------------------------");

    if final_errors_count > 0 {
        Err(anyhow!("Processing finished with {} errors.", final_errors_count))
    } else {
        Ok(())
    }
}