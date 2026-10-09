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
/// The delimiter is `,` and no header row is written unless set with
/// [`set_separator`](Self::set_separator) and [`set_header`](Self::set_header).
/// Parent directories are created as needed; the buffer is flushed every 100 rows and on drop.
///
/// ```no_run
/// use dsmc::logger::DataStorage;
///
/// let mut log = DataStorage::new("./out/log.csv").unwrap().set_header(["t", "y"]);
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
    /// Create (overwrite) the CSV file at `path`.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let path = path.as_ref();
        if let Some(parent) = path.parent().filter(|parent| !parent.exists()) {
            fs::create_dir_all(parent)?;
        }

        let file = File::create(path)?;
        let buf_writer = BufWriter::new(file);

        Ok(Self {
            writer: Some(Self::csv_writer(buf_writer, b',')),
            cnt: 0,
        })
    }

    fn csv_writer(buf_writer: BufWriter<File>, delimiter: u8) -> csv::Writer<BufWriter<File>> {
        WriterBuilder::new()
            .has_headers(false)
            .delimiter(delimiter)
            .from_writer(buf_writer)
    }

    /// Use `separator` as the delimiter (ignored if not ASCII). Call before `set_header` and `add`;
    /// rows already written keep the previous delimiter.
    pub fn set_separator(mut self, separator: char) -> Self {
        if !separator.is_ascii() {
            return self;
        }
        if let Some(writer) = self.writer.take() {
            self.writer = Some(match writer.into_inner() {
                Ok(buf_writer) => Self::csv_writer(buf_writer, separator as u8),
                Err(err) => err.into_inner(),
            });
        }
        self
    }

    /// Write `header` as a row now. Call before the first `add`. The row only goes into the write
    /// buffer, so I/O errors appear at the next flush (`add` every 100 rows, `flush` or `close`).
    /// `header` can be any iterable of strings: `["t", "y"]`, `&[&str]`, `Vec<String>`, `&Vec<String>`,
    /// an iterator, etc.
    pub fn set_header<I, T>(mut self, header: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: AsRef<str>,
    {
        if let Some(writer) = &mut self.writer {
            // write fields one by one (an owned `String` item cannot lend its bytes to `write_record`),
            // then end the row with an empty record
            for name in header {
                let _ = writer.write_field(name.as_ref());
            }
            let _ = writer.write_record(None::<&[u8]>);
        }
        self
    }

    /// Write `data` as one row (a struct, tuple, array or slice of values).
    pub fn add<T: Serialize>(&mut self, data: &T) -> Result<(), Box<dyn Error>> {
        if let Some(writer) = &mut self.writer {
            writer.serialize(data)?;
            self.cnt += 1;

            // `is_multiple_of` needs Rust 1.87; edition 2024 allows 1.85
            #[allow(clippy::manual_is_multiple_of)]
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
