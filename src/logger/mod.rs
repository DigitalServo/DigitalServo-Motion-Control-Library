//! CSV logging of `Serialize` values.

use csv::WriterBuilder;
use serde::Serialize;
use std::error::Error;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;

pub mod serializer;

/// CSV logger: each `add` writes one `Serialize` value as a row (the fields of a struct, or the
/// elements of a tuple, array or slice, become the columns). Number formats of struct fields can be
/// fixed with the [`serializer`] functions.
/// Parent directories are created as needed; the buffer is flushed every 100 rows and on drop.
///
/// ```no_run
/// use dsmc::logger::DataStorage;
///
/// let mut log = DataStorage::new("./out/log.csv", ',', false).unwrap();
/// for k in 0..100 {
///     let t = k as f64 * 1e-3;
///     log.add(&[t, t.sin()]).unwrap();
/// }
/// log.close().unwrap();
/// ```
pub struct DataStorage {
    writer: Option<csv::Writer<BufWriter<File>>>,
    cnt: usize,
}

impl DataStorage {
    /// Create (overwrite) the CSV file at `path`. `separator` is the delimiter (`,` if not ASCII);
    /// `has_header` writes struct field names as the first row.
    pub fn new<P: AsRef<Path>>(path: P, separator: char, has_header: bool) -> Result<Self, Box<dyn Error>> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent)?;
            }
        }

        let delimiter: u8 = match separator.is_ascii() {
            true => separator as u8,
            false => b',',
        };

        let file = File::create(path)?;
        let buf_writer = BufWriter::new(file);

        let csv_writer = WriterBuilder::new()
            .has_headers(has_header)
            .delimiter(delimiter)
            .from_writer(buf_writer);

        Ok(Self {
            writer: Some(csv_writer),
            cnt: 0,
        })
    }

    /// Write `data` as one row (a struct, tuple, array or slice of values).
    pub fn add<T: Serialize>(&mut self, data: &T) -> Result<(), Box<dyn Error>> {
        if let Some(writer) = &mut self.writer {
            writer.serialize(data)?;
            self.cnt += 1;

            if self.cnt % 100 == 0 {
                let _ = writer.flush();
            }
        }
        Ok(())
    }

    /// Flush buffered rows to the file.
    pub fn flush(&mut self) -> Result<(), Box<dyn Error>> {
        if let Some(writer) = &mut self.writer {
            writer.flush()?;
        }
        Ok(())
    }

    /// Flush and close the file.
    pub fn close(mut self) -> Result<(), Box<dyn Error>> {
        if let Some(mut writer) = self.writer.take() {
            writer.flush()?;
        }
        Ok(())
    }
}

impl Drop for DataStorage {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}
