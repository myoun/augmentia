//! Flush-on-event JSONL output. Existing artifacts are never silently replaced.

use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;

pub struct JsonlWriter<W: Write> {
    writer: W,
    failed: bool,
}

impl JsonlWriter<BufWriter<File>> {
    pub fn create_new(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        Ok(Self::new(BufWriter::new(file)))
    }
}

impl<W: Write> JsonlWriter<W> {
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            failed: false,
        }
    }

    /// A successful return means the complete line reached the underlying writer and was flushed.
    /// This is process-interruption durability, not a promise of power-loss durability (`fsync`).
    pub fn write<T: Serialize>(&mut self, value: &T) -> io::Result<()> {
        if self.failed {
            return Err(io::Error::other(
                "JSONL writer failed previously; refusing to continue",
            ));
        }
        let result = (|| {
            let mut line = serde_json::to_vec(value).map_err(io::Error::other)?;
            line.push(b'\n');
            self.writer.write_all(&line)?;
            self.writer.flush()
        })();
        self.failed = result.is_err();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Failure {
        flush: bool,
    }
    impl Write for Failure {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.flush {
                Ok(bytes.len())
            } else {
                Err(io::Error::other("write failed"))
            }
        }
        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::other("flush failed"))
        }
    }

    #[test]
    fn write_and_flush_errors_are_propagated_and_poison_the_writer() {
        for flush in [false, true] {
            let mut writer = JsonlWriter::new(Failure { flush });
            let error = writer.write(&42).unwrap_err();
            assert_eq!(
                error.to_string(),
                if flush {
                    "flush failed"
                } else {
                    "write failed"
                }
            );
            assert!(
                writer
                    .write(&43)
                    .unwrap_err()
                    .to_string()
                    .contains("previously")
            );
        }
    }
}
