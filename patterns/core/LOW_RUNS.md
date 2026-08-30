# Low Runs — data source and derivation

`low_runs.csv` drives the `LOW_RUN_6M` and `LOW_RUN_12M` patterns. This note
records **how the data is derived** so the list can be maintained (and, one day,
generated automatically) instead of trusted blindly.

## What a "low run" is

A **low run** is a district/block combination where a **single BEP facility**
printed an unusually small slice of that block — **6.4 million** or **12.8
million** notes — instead of a normal large run. Fewer notes were made, so
collectors value them.

## The source chart

Print data comes from the per-series serial charts at **uspapermoney.io**, e.g.
Series 2021 $1: <https://www.uspapermoney.io/serials/f2021_s.html>

The chart is a **monthly grid**:

| element | meaning |
|---|---|
| rows | production months (e.g. Dec 2022 … present) |
| columns A–L | the 12 Federal Reserve districts |
| cell | the serial range printed that district that month, shown as `start … end` |
| suffix letter in a serial | the **block** (e.g. `I 896 00000 A` → block `A`) |
| cell color | the **facility**: Fort Worth (FW) vs Washington DC |

## The key rules for reading it

1. **Implied 96M cap.** Each block runs serial `00000001 → 96000000`, then
   production rolls over to the next block letter. The chart does **not** always
   print the `96 000 000` boundary or the `00 000 001` restart — you are expected
   to know the cap is there. So a cell like `I 896 00000 A … I 448 00000 B` means:
   finish block A from 89.6M up to the 96M cap (**6.4M notes**), then start block
   B from 1 to 44.8M.

2. **Low run = a facility's slice of one block.** Follow each district's serial
   count across the months, attributing each month's increment to that month's
   facility, and split at every 96M block boundary. If a facility's total for a
   single block is exactly **6.4M** or **12.8M**, that's a low run.
   *Example:* District I block A — DC printed 1 → 89.6M, then Fort Worth finished
   it 89.6M → 96M = **6.4M at FW** → `2021,I,A,FW,6.4`.

3. **Only count finished blocks.** A block that currently sits at 6.4M/12.8M but
   is **still being printed in the latest month** is not a low run yet — it will
   keep growing. Wait until production has moved on before adding it.

4. **Re-check on updates.** As the BEP keeps printing, a block that was a low run
   can grow past 6.4M/12.8M and stop being one. (2021 example: `K/D/FW` was a 6.4M
   low run, later grew to 83.2M, and was removed.) Conversely new small slices
   appear. Re-derive from the current chart when the site updates.

## CSV format

```
series,district,block,facility,quantity
2021,I,A,FW,6.4
```
- `district` = first letter of the serial; `block` = last letter (suffix)
- `facility` = `FW` (Fort Worth, front plate starts with `FW`) or `DC`
- `quantity` = `6.4` or `12.8` (millions)

## Maintenance / future automation

`tools/parse_low_runs.py` implements rules 1–3 above against a saved copy of a
uspapermoney chart and prints candidate `low_runs.csv` rows (flagging still-open
blocks so they aren't added prematurely). It is a standalone helper today; a
future feature could fetch the site directly and diff against `low_runs.csv` on
each update. See that script's header for usage.
