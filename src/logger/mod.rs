use csv::WriterBuilder;
use serde::Serialize;
use std::error::Error;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;

pub mod serializer;

pub struct DataStorage {
    writer: Option<csv::Writer<BufWriter<File>>>,
    cnt: usize,
}

impl DataStorage {
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

    pub fn flush(&mut self) -> Result<(), Box<dyn Error>> {
        if let Some(writer) = &mut self.writer {
            writer.flush()?;
        }
        Ok(())
    }

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
