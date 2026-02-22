/// Data loader module supporting multiple file formats
use crate::models::{LotteryDraw, LotteryDataset};
use anyhow::{Context, Result};
use calamine::{open_workbook, Reader, Xlsx, Xls};
use chrono::{DateTime, Utc, TimeZone};
use csv::ReaderBuilder;
use rusqlite::Connection;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub struct DataLoader;

impl DataLoader {
    /// Load data from any supported format
    pub fn load_from_file(path: &Path) -> Result<LotteryDataset> {
        let extension = path
            .extension()
            .and_then(|s| s.to_str())
            .context("Invalid file extension")?
            .to_lowercase();

        match extension.as_str() {
            "csv" => Self::load_from_csv(path),
            "xlsx" => Self::load_from_xlsx(path),
            "xls" => Self::load_from_xls(path),
            "dat" => Self::load_from_dat(path),
            "db" | "sqlite" | "sqlite3" => Self::load_from_sqlite(path),
            _ => Err(anyhow::anyhow!("Unsupported file format: {}", extension)),
        }
    }

    /// Load from CSV file
    /// Expected format: date,number1,number2,number3,...
    fn load_from_csv(path: &Path) -> Result<LotteryDataset> {
        let file = File::open(path)?;
        let mut rdr = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);

        let mut draws = Vec::new();
        let mut min_number = u32::MAX;
        let mut max_number = 0u32;
        let mut numbers_per_draw = 0;

        for result in rdr.records() {
            let record = result?;

            // Parse date (first column)
            let date_str = record.get(0).context("Missing date column")?;
            let draw_date = Self::parse_date(date_str)?;

            // Parse numbers (remaining columns)
            let mut numbers = Vec::new();
            for i in 1..record.len() {
                if let Some(num_str) = record.get(i) {
                    if !num_str.trim().is_empty() {
                        let num: u32 = num_str.trim().parse()?;
                        numbers.push(num);
                        min_number = min_number.min(num);
                        max_number = max_number.max(num);
                    }
                }
            }

            if numbers_per_draw == 0 {
                numbers_per_draw = numbers.len();
            }

            draws.push(LotteryDraw {
                id: Some(draws.len() as u64),
                numbers,
                draw_date,
                game_name: path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("Unknown")
                    .to_string(),
            });
        }

        Ok(LotteryDataset {
            draws,
            min_number,
            max_number,
            numbers_per_draw,
        })
    }

    /// Load from Excel XLSX file
    fn load_from_xlsx(path: &Path) -> Result<LotteryDataset> {
        let mut workbook: Xlsx<_> = open_workbook(path)?;

        // Get first worksheet
        let sheet_name = workbook.sheet_names()[0].clone();
        let range = workbook
            .worksheet_range(&sheet_name)
            .context("Cannot find worksheet")?;

        Self::parse_excel_range(&range, path)
    }

    /// Load from Excel XLS file
    fn load_from_xls(path: &Path) -> Result<LotteryDataset> {
        let mut workbook: Xls<_> = open_workbook(path)?;

        // Get first worksheet
        let sheet_name = workbook.sheet_names()[0].clone();
        let range = workbook
            .worksheet_range(&sheet_name)
            .context("Cannot find worksheet")?;

        Self::parse_excel_range(&range, path)
    }

    /// Parse Excel range (common for XLS and XLSX)
    fn parse_excel_range(
        range: &calamine::Range<calamine::DataType>,
        path: &Path,
    ) -> Result<LotteryDataset> {
        let mut draws = Vec::new();
        let mut min_number = u32::MAX;
        let mut max_number = 0u32;
        let mut numbers_per_draw = 0;

        // Skip header row
        for (row_idx, row) in range.rows().skip(1).enumerate() {
            // First column: date
            let date_str = row[0].to_string();
            let draw_date = Self::parse_date(&date_str)?;

            // Remaining columns: numbers
            let mut numbers = Vec::new();
            for cell in row.iter().skip(1) {
                if let Ok(num) = cell.to_string().trim().parse::<u32>() {
                    numbers.push(num);
                    min_number = min_number.min(num);
                    max_number = max_number.max(num);
                }
            }

            if numbers_per_draw == 0 {
                numbers_per_draw = numbers.len();
            }

            if !numbers.is_empty() {
                draws.push(LotteryDraw {
                    id: Some(row_idx as u64),
                    numbers,
                    draw_date,
                    game_name: path.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("Unknown")
                        .to_string(),
                });
            }
        }

        Ok(LotteryDataset {
            draws,
            min_number,
            max_number,
            numbers_per_draw,
        })
    }

    /// Load from DAT file (space or tab separated)
    fn load_from_dat(path: &Path) -> Result<LotteryDataset> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut draws = Vec::new();
        let mut min_number = u32::MAX;
        let mut max_number = 0u32;
        let mut numbers_per_draw = 0;

        for (idx, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            // Try to parse first field as date, otherwise use current time
            let (date_offset, draw_date) = if let Ok(date) = Self::parse_date(parts[0]) {
                (1, date)
            } else {
                (0, Utc::now())
            };

            let mut numbers = Vec::new();
            for part in parts.iter().skip(date_offset) {
                if let Ok(num) = part.parse::<u32>() {
                    numbers.push(num);
                    min_number = min_number.min(num);
                    max_number = max_number.max(num);
                }
            }

            if numbers_per_draw == 0 {
                numbers_per_draw = numbers.len();
            }

            if !numbers.is_empty() {
                draws.push(LotteryDraw {
                    id: Some(idx as u64),
                    numbers,
                    draw_date,
                    game_name: path.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("Unknown")
                        .to_string(),
                });
            }
        }

        Ok(LotteryDataset {
            draws,
            min_number,
            max_number,
            numbers_per_draw,
        })
    }

    /// Load from SQLite database
    /// Expected schema: CREATE TABLE draws (id INTEGER PRIMARY KEY, date TEXT, numbers TEXT)
    fn load_from_sqlite(path: &Path) -> Result<LotteryDataset> {
        let conn = Connection::open(path)?;

        let mut stmt = conn.prepare(
            "SELECT id, date, numbers FROM draws ORDER BY date ASC"
        )?;

        let mut draws = Vec::new();
        let mut min_number = u32::MAX;
        let mut max_number = 0u32;
        let mut numbers_per_draw = 0;

        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;

        for row in rows {
            let (id, date_str, numbers_str) = row?;
            let draw_date = Self::parse_date(&date_str)?;

            // Parse numbers (expected format: "1,2,3,4,5" or "1 2 3 4 5")
            let numbers: Vec<u32> = numbers_str
                .split(|c| c == ',' || c == ' ')
                .filter_map(|s| s.trim().parse::<u32>().ok())
                .collect();

            for &num in &numbers {
                min_number = min_number.min(num);
                max_number = max_number.max(num);
            }

            if numbers_per_draw == 0 {
                numbers_per_draw = numbers.len();
            }

            draws.push(LotteryDraw {
                id: Some(id as u64),
                numbers,
                draw_date,
                game_name: path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("Unknown")
                    .to_string(),
            });
        }

        Ok(LotteryDataset {
            draws,
            min_number,
            max_number,
            numbers_per_draw,
        })
    }

    /// Parse date from string (supports multiple formats)
    fn parse_date(date_str: &str) -> Result<DateTime<Utc>> {
        // Try various date formats
        let formats = vec![
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%d-%m-%Y %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
        ];

        for format in formats {
            if let Ok(date) = chrono::NaiveDateTime::parse_from_str(date_str, format) {
                return Ok(Utc.from_utc_datetime(&date));
            }
            if let Ok(date) = chrono::NaiveDate::parse_from_str(date_str, format) {
                return Ok(Utc.from_utc_datetime(&date.and_hms_opt(0, 0, 0).unwrap()));
            }
        }

        Err(anyhow::anyhow!("Cannot parse date: {}", date_str))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_date_parsing() {
        assert!(DataLoader::parse_date("2025-10-13").is_ok());
        assert!(DataLoader::parse_date("13/10/2025").is_ok());
        assert!(DataLoader::parse_date("2025-10-13 14:30:00").is_ok());
    }
}
