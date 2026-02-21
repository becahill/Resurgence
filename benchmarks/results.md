# Benchmarks

Generated at: 2026-02-21T23:02:52.838371+00:00
Host: macOS-26.3-arm64-arm-64bit-Mach-O
Python: 3.13.7

Configuration:
- Path counts: 1k, 10k, 100k, 1.0M
- Horizon days: 10
- Sample points: 2500
- Seed: 7

| Paths | Python runtime (s) | Python peak RSS (MB) | Rust runtime (s) | Rust peak RSS (MB) | Speedup (x) |
|---:|---:|---:|---:|---:|---:|
| 1k | 0.0007 | 48.52 | 0.0003 | 48.61 | 2.39 |
| 10k | 0.0038 | 48.77 | 0.0010 | 48.55 | 3.86 |
| 100k | 0.0319 | 57.39 | 0.0067 | 54.83 | 4.80 |
| 1.0M | 0.3176 | 161.48 | 0.0679 | 109.73 | 4.67 |

Raw machine-readable output: `benchmarks/results.json`.
