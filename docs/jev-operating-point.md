# Jev operating point — held-out validation & calibration

## Held-out threshold selection (split by recording)

| tune half | best th (tune F1) | held-out P | R | F1 |
|---|---|---|---|---|
| even records | 0.2 (0.803) | 0.744 | 0.847 | 0.792 |
| odd records | 0.15 (0.797) | 0.700 | 0.917 | 0.794 |

All pairs at threshold 0.3: P=0.779 R=0.778 F1=0.779; POLKE pipeline on the same pairs: F1=0.743.

## Calibration (all pairs)

| bin | n | mean p | empirical positive rate |
|---|---|---|---|
| 0.0-0.1 | 208 | 0.063 | 0.293 |
| 0.1-0.2 | 244 | 0.143 | 0.357 |
| 0.2-0.3 | 215 | 0.242 | 0.512 |
| 0.3-0.4 | 178 | 0.346 | 0.573 |
| 0.4-0.5 | 138 | 0.442 | 0.645 |
| 0.5-0.6 | 125 | 0.550 | 0.672 |
| 0.6-0.7 | 112 | 0.647 | 0.795 |
| 0.7-0.8 | 150 | 0.748 | 0.840 |
| 0.8-0.9 | 170 | 0.845 | 0.865 |
| 0.9-1.0 | 253 | 0.940 | 0.949 |

ECE = 0.156. Probability mass Sum(p) = 853 vs 1127 actual gold positives (0.76x) over 1779 pairs.

## Per-construct reliability at threshold 0.17

170 constructs OK (F1 >= 0.5), 23 weak, 413 with < 4 judged pairs (not assessable).

Weak constructs (flag list):

- NCL-02: F1 0.00 (4 pairs, 1 gold positive)
- NCL-04: F1 0.00 (4 pairs, 0 gold positive)
- NCL-08: F1 0.00 (4 pairs, 1 gold positive)
- NFV-25: F1 0.00 (4 pairs, 0 gold positive)
- PHV-02: F1 0.00 (4 pairs, 3 gold positive)
- PREP-25: F1 0.00 (4 pairs, 0 gold positive)
- QIN-02: F1 0.00 (4 pairs, 1 gold positive)
- VCP-27: F1 0.00 (4 pairs, 0 gold positive)
- VCP-33: F1 0.00 (4 pairs, 0 gold positive)
- VTA-02: F1 0.00 (4 pairs, 0 gold positive)
- NCL-13: F1 0.40 (4 pairs, 3 gold positive)
- NEG-12: F1 0.40 (4 pairs, 2 gold positive)
- NFV-07: F1 0.40 (4 pairs, 1 gold positive)
- NOU-18: F1 0.40 (4 pairs, 1 gold positive)
- PREP-22: F1 0.40 (4 pairs, 2 gold positive)
- PREP-24: F1 0.40 (4 pairs, 1 gold positive)
- QUE-07: F1 0.40 (4 pairs, 1 gold positive)
- REL-07: F1 0.40 (4 pairs, 1 gold positive)
- REL-11: F1 0.40 (4 pairs, 3 gold positive)
- REL-13: F1 0.40 (4 pairs, 1 gold positive)
- REP-01: F1 0.40 (4 pairs, 4 gold positive)
- REP-11: F1 0.40 (4 pairs, 1 gold positive)
- VTA-05: F1 0.40 (4 pairs, 2 gold positive)
