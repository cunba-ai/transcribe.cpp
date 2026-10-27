//! Minimal RIFF/WAVE reader for the e2e samples: PCM 8/16/24/32-bit and
//! IEEE float 32/64, mono or stereo (stereo is averaged). The samples in
//! `samples/` are 16 kHz mono 16-bit; the loader asserts the rate callers
//! need and converts amplitudes to f32 in [-1, 1].

#[derive(Debug)]
pub enum WavError {
    Io(String),
    Format(String),
}

impl std::fmt::Display for WavError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WavError::Io(e) => write!(f, "io: {e}"),
            WavError::Format(e) => write!(f, "format: {e}"),
        }
    }
}

pub struct Wav {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}

pub fn load(path: &std::path::Path) -> Result<Wav, WavError> {
    let bytes = std::fs::read(path).map_err(|e| WavError::Io(format!("{path:?}: {e}")))?;
    let mut r = Reader { b: &bytes, pos: 0 };

    let magic = r.take_str(4)?;
    if magic != "RIFF" {
        return Err(WavError::Format("not a RIFF file (no 'RIFF' magic)".into()));
    }
    let _riff_size = r.take_u32()?;
    let wave = r.take_str(4)?;
    if wave != "WAVE" {
        return Err(WavError::Format("RIFF type is not 'WAVE'".into()));
    }

    let mut fmt: Option<Fmt> = None;
    let mut data: Option<(usize, usize)> = None; // (offset, len)
    while r.pos + 8 <= r.b.len() {
        let id = r.take_str(4)?;
        let size = r.take_u32()? as usize;
        let body = r.pos;
        match &id[..] {
            "fmt " => {
                let audio_format = slice_u16(r.b, body)?;
                let channels = slice_u16(r.b, body + 2)?;
                let sample_rate = slice_u32(r.b, body + 4)?;
                let bits = slice_u16(r.b, body + 14)?;
                fmt = Some(Fmt { audio_format, channels, sample_rate, bits });
            }
            "data" => {
                let end = (body + size).min(r.b.len());
                data = Some((body, end - body));
            }
            _ => {}
        }
        // chunks are word-aligned
        r.pos = body + size + (size & 1);
    }

    let fmt = fmt.ok_or_else(|| WavError::Format("missing 'fmt ' chunk".into()))?;
    if fmt.channels == 0 || !matches!(fmt.bits, 8 | 16 | 24 | 32 | 64) {
        return Err(WavError::Format(format!(
            "bad fmt: {} channels, {} bits",
            fmt.channels, fmt.bits
        )));
    }
    let (off, len) = data.ok_or_else(|| WavError::Format("missing 'data' chunk".into()))?;
    let raw = &r.b[off..off + len];

    let bytes_per = (fmt.bits / 8) as usize;
    let frames = raw.len() / (fmt.channels as usize * bytes_per);
    let mut samples = Vec::with_capacity(frames);
    let ch = fmt.channels as usize;
    match (fmt.audio_format, fmt.bits) {
        (1, 8) => {
            for i in 0..frames {
                samples.push(ch_avg((0..ch).map(|c| raw[i * ch + c] as f32 / 128.0 - 1.0)));
            }
        }
        (1, 16) => {
            for i in 0..frames {
                samples.push(ch_avg(
                    (0..ch).map(|c| i16::from_le_bytes([raw[(i * ch + c) * 2], raw[(i * ch + c) * 2 + 1]]) as f32 / 32768.0),
                ));
            }
        }
        (1, 24) => {
            for i in 0..frames {
                samples.push(ch_avg((0..ch).map(|c| {
                    let b = (i * ch + c) * 3;
                    let v = ((raw[b] as i32) << 8 | (raw[b + 1] as i32) << 16 | (raw[b + 2] as i32) << 24) >> 8;
                    v as f32 / 8388608.0
                })));
            }
        }
        (1, 32) => {
            for i in 0..frames {
                samples.push(ch_avg((0..ch).map(|c| {
                    let b = (i * ch + c) * 4;
                    i32::from_le_bytes([raw[b], raw[b + 1], raw[b + 2], raw[b + 3]]) as f32 / 2147483648.0
                })));
            }
        }
        (3, 32) => {
            for i in 0..frames {
                samples.push(ch_avg((0..ch).map(|c| {
                    let b = (i * ch + c) * 4;
                    f32::from_le_bytes([raw[b], raw[b + 1], raw[b + 2], raw[b + 3]])
                })));
            }
        }
        (3, 64) => {
            for i in 0..frames {
                samples.push(ch_avg((0..ch).map(|c| {
                    let b = (i * ch + c) * 8;
                    f64::from_le_bytes(raw[b..b + 8].try_into().unwrap()) as f32
                })));
            }
        }
        (f, b) => return Err(WavError::Format(format!("unsupported format {f}, {b} bits"))),
    }
    Ok(Wav { sample_rate: fmt.sample_rate, channels: fmt.channels, samples })
}

struct Fmt {
    audio_format: u16,
    channels: u16,
    sample_rate: u32,
    bits: u16,
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take_str(&mut self, n: usize) -> Result<&'a str, WavError> {
        if self.pos + n > self.b.len() {
            return Err(WavError::Format("truncated header".into()));
        }
        let s = std::str::from_utf8(&self.b[self.pos..self.pos + n])
            .map_err(|_| WavError::Format("non-ascii chunk id".into()))?;
        self.pos += n;
        Ok(s)
    }
    fn take_u32(&mut self) -> Result<u32, WavError> {
        let v = slice_u32(self.b, self.pos)?;
        self.pos += 4;
        Ok(v)
    }
}

fn slice_u16(b: &[u8], pos: usize) -> Result<u16, WavError> {
    b.get(pos..pos + 2)
        .map(|s| u16::from_le_bytes([s[0], s[1]]))
        .ok_or_else(|| WavError::Format("truncated u16".into()))
}
fn slice_u32(b: &[u8], pos: usize) -> Result<u32, WavError> {
    b.get(pos..pos + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        .ok_or_else(|| WavError::Format("truncated u32".into()))
}

fn ch_avg(it: impl Iterator<Item = f32>) -> f32 {
    let mut sum = 0.0;
    let mut n = 0;
    for v in it {
        sum += v;
        n += 1;
    }
    sum / n as f32
}
