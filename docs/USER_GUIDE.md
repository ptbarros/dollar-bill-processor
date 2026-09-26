# Dollar Detective — User Guide

A practical guide to using Dollar Detective: preparing scans, processing them,
reading results, and managing patterns.

> This guide is a work in progress. Sections marked _(to be expanded)_ are
> stubs — the workflow is described in one line so the outline is complete, and
> the detail will be filled in over time.

## Contents

1. [Preparing a folder: Organize vs. Verify](#preparing-a-folder-organize-vs-verify)
2. [Folder formats the app recognizes](#folder-formats-the-app-recognizes)
3. [The two modes: Manual vs. Scan](#the-two-modes-manual-vs-scan)
4. [Processing a folder (Manual mode)](#processing-a-folder-manual-mode) _(to be expanded)_
5. [Scanning straps live (Scan mode)](#scanning-straps-live-scan-mode)
6. [Reading the results & overlays](#reading-the-results--overlays) _(to be expanded)_
7. [How odds & rarity work](#how-odds--rarity-work)
8. [Pattern Manager](#pattern-manager) _(to be expanded)_
9. [Serial Lookup & Strap Check](#serial-lookup--strap-check) _(to be expanded)_
10. [Physical-print detections](#physical-print-detections-gas-pump-seal-shift-plates) _(to be expanded)_
11. [Cropping for listings](#cropping-for-listings) _(to be expanded)_
12. [Labels](#labels) _(to be expanded)_
13. [Insights report](#insights-report) _(to be expanded)_
14. [Backup & Restore](#backup--restore) _(to be expanded)_

---

## Preparing a folder: Organize vs. Verify

These two features overlap on one job — getting the **front (serial side)** and
**back** of each bill sorted correctly — but they are very different tools. The
short version:

- **Verify front/back pairs** is a *setting* that decides front vs. back **in
  memory, each time you process**. It never changes your files.
- **Organize Folder** is a *one-time action* that **rewrites the files on disk**
  — it sorts front/back **and** fixes orientation, deskews, and renames them.

### Verify front/back pairs (a setting)

Found in **Settings → Processing**. It controls *how hard the app works to figure
out which image in each pair is the front* during a run. Its only job is
front/back assignment; it does not rotate, straighten, or rename anything.

- **Checked (ON):** before processing begins, the app runs one detection pass
  over every image, classifies front vs. back, **swaps the pair if it's
  backwards**, and caches what it found for the rest of the run. Slower to
  *start*, but every bill then runs at a steady speed.
  **Best for:** unsorted piles with random order/orientation.

- **Unchecked (OFF):** the app skips that upfront pass and works lazily — it
  assumes each pair is already in order, reads the serial off the front, and
  **only swaps if it finds no serial** (and there's a back to fall back on).
  Fast start; you only pay the swap cost when a pair is actually wrong.
  **Best for:** scanner output that's usually already in the right order.

Either way, nothing is written to disk — it's a per-run behavior.

### Organize Folder (Edit → Organize Folder…)

A heavier, **permanent** prep step you run **once on a folder before
processing**, from the **Edit → Organize Folder…** menu. It does everything
Verify does **and more**, and saves the results to disk. For each pair it:

- Classifies front vs. back
- **Fixes upside-down orientation**
- **Corrects skew** (straightens using the serial characters)
- **Renames the files to `Dollar_NNN.jpg`** — odd numbers = fronts, even = backs —
  and deletes the originals

So it bakes correct pairing, orientation, and rotation *into the files
themselves*.

> ⚠️ **Organize overwrites and renames your originals in place.** Run it on a
> copy (or on scanner output you don't mind renaming), not on your only copy of
> a set.

### How they interact — the key part

Once a folder has been Organized, its files are in the `Dollar_NNN`
("pre-organized") format. On the next run, the app **automatically skips both the
Verify step *and* the alignment step**, no matter how the checkbox is set —
because all of that work is already done and saved in the files. That's what
makes repeat runs of an organized folder fast and consistent.

**So:** if you Organize a folder, the Verify setting no longer matters for it.
Verify only comes into play for folders you feed in **as-is**, without organizing
first.

### At a glance

| | **Verify front/back pairs** | **Organize Folder** |
|---|---|---|
| What it is | A checkbox (per-run behavior) | A menu action, **Edit → Organize Folder…** (one-time) |
| Changes files on disk? | No — memory only | **Yes** — rotates, deskews, renames, deletes originals |
| Scope | Front/back assignment only | Front/back **+ orientation + skew + renaming** |
| When it runs | Start of each run (ON), or lazily during it (OFF) | When you run it from the menu |
| Effect on later runs | None persists | Folder becomes pre-organized → Verify **and** alignment skipped, so runs are faster |
| Best for | Choosing a front/back detection strategy per scan style | Prepping a folder once for fast, repeatable processing |

### Which should I use?

- **Just processing a folder once, as-is?** Leave it to the **Verify** setting —
  ON for messy piles, OFF for tidy scanner output.
- **Going to process the same folder repeatedly, or want the fastest runs?**
  **Organize** it once (on a copy), then process.

---

## Folder formats the app recognizes

The app auto-detects how a folder is laid out and picks the fastest safe path:

- **`Dollar_NNN.jpg` (pre-organized)** — the output of **Organize Folder**
  (odd = front, even = back). Fastest path: Verify and alignment are skipped.
- **Suffix pairs** — e.g. `0001.jpg` + `0001_b.jpg` (the `_b` marks the back).
- **Sequential** — alternating numbered images that pair up in order.

This detection is why an Organized folder processes faster than a raw one.

---

## The two modes: Manual vs. Scan

The toggle at the **top-left of the toolbar** switches the app between its two
ways of working. Clicking **Manual** or **Scan** reconfigures the toolbar for
that workflow — the rest of the app (results list, preview, overlays, patterns)
is identical in both.

Both modes end the same way: they produce a **strap** — a numbered folder that
holds a run of bills, their fancy crops, and a `results.csv`. The only
difference is where the bills *come from*.

### Manual mode — "I already have the images"

Use this when the scans already exist in a folder (you scanned earlier, someone
sent you a folder, etc.).

1. Set **Input** to the folder of scans (and **Output**, where fancy crops go —
   it auto-fills to a `fancy_bills` subfolder).
2. Click **Process**. Each bill's serial is read, classified against the enabled
   patterns, and flagged fancy or sent to review. The results fill in as it runs.
3. Review the results.
4. When you're happy with the run, click **File Strap** to file it as the next
   strap in your sequence (see below). This is optional — if you just wanted to
   check a folder, you don't have to file it.

### Scan mode — "scan straps as I go"

Use this to let the app **watch your scanner's output folder** and file each
strap for you, without pointing it at a folder by hand.

1. Click **Start Scanning**. The toolbar shows a **Watching:** indicator with the
   folder it's watching and the next strap number.
2. Feed the strap on your scanner (in as many passes as you like — a pause never
   ends the strap).
3. Click **Stop & File Batch**. Everything that arrived is filed as the next
   strap and processed.

> **First-time setup:** the first time you enter Scan mode, a short **setup
> wizard** offers to walk you through the folders and strap naming it needs.
> You can re-run it any time from **Help → Setup Wizard**, or set the same
> values directly under **Settings → Folders**.

See [Scanning straps live (Scan mode)](#scanning-straps-live-scan-mode) for the
optional live-processing option.

### Straps are one shared, numbered sequence

However a strap is created — **File Strap** in Manual mode or **Stop & File
Batch** in Scan mode — it goes into the **same Straps folder** and takes the
**next number** in one shared sequence (`001`, `002`, `003`, …). So a manually
filed strap and a scanned strap are indistinguishable afterward: both appear in
the **batch dropdown** above the results list, and selecting one reopens it.

A filed strap is **self-contained** — the bill images, a `fancy_bills` subfolder
of crops, and a `results.csv` all live inside that strap's folder. That's what
lets the batch dropdown list and reopen it later.

> **Where things live:** set the **Straps Folder** (and the scanner's output
> folder it watches) on **Settings → Folders**. By default, filing a strap
> **moves** the bills into it; turn on **"Keep originals when filing a strap"**
> (Settings → Processing) if your input folder is a library you don't want
> emptied.

| | **Manual mode** | **Scan mode** |
|---|---|---|
| Bills come from | A folder you already have | Your scanner's output folder, watched live |
| You click | **Process**, then optionally **File Strap** | **Start Scanning** → **Stop & File Batch** |
| Filing is | Optional (only if you want to keep the run as a strap) | The normal end of a scan |
| Result | The next strap in the sequence | The next strap in the sequence |
| Best for | Re-checking existing scans, folders sent to you | Working straps live at the scanner |

---

## Processing a folder (Manual mode)
_(to be expanded)_ — Point the app at a folder, start processing, and watch the
results fill in. Each bill's serial is read, classified against the enabled
patterns, and flagged fancy or sent to review. When you're done, **File Strap**
files the run as the next strap (see [The two modes](#the-two-modes-manual-vs-scan)).

## Scanning straps live (Scan mode)

Instead of scanning to a folder and then pointing the app at it, you can let the
app **watch your scanner's output folder** and file each strap for you. Switch to
**Scan** mode, click **Start Scanning**, feed the strap (in as many passes as you
like), then click **Stop & File Batch** — everything that arrived is moved into a
new strap under your **Straps** folder and processed. Past straps stay in the
batch dropdown so you can reopen them.

### Live processing (experimental) — depends on how your scanner saves files

There's an experimental **Process live** checkbox that starts classifying scans
*as they arrive* instead of waiting until you click Stop. Whether it actually
saves you time depends entirely on **when your scanner writes the image files**:

- **If the scanner writes each page (or each pass) to the folder as it feeds**,
  live processing works through them while you keep scanning.
- **If the scanner holds the whole strap in memory and writes every file at once
  at the end**, there's nothing to process until you finish, so live mode just
  processes the batch at the end — the same as leaving the box unchecked. No
  harm, but no time saved.

> **Canon imageFormula R40:** this scanner buffers all pages in the Canon
> software and only writes them to the output folder when you click **Finish**
> in its interface — so every file appears at once. **Live processing gives no
> benefit on the R40**; just scan the strap, then click Stop & File Batch as
> normal. (Behavior varies by scanner and by its "save" settings — check yours
> by watching the output folder while you feed a strap: do files trickle in, or
> appear all at once at the end?)

## Reading the results & overlays
_(to be expanded)_ — The results list, the fancy/kept columns, and the on-bill
overlay that boxes/connects the digits a pattern matched. Overlay colors are
assigned by first-appearance from a fixed high-contrast rotation (so a
single-color pattern draws in one color; only patterns that define distinct
sub-groups show multiple colors).

## How odds & rarity work

Every pattern shows odds like **"1 in 1,901"**. Here's what that number is, how
it's produced, and — importantly — what it does *not* tell you.

### The odds are an exact count, not an estimate

The odds are computed by brute force: the pattern's real matching rule is run
against **all 96,000,000 printable serials**\* and the matches are counted. If a
pattern matches 50,495 of them, its odds are 96,000,000 ÷ 50,495 = **1 in 1,901**.
No sampling and no formula — a literal count of every serial that qualifies.

> \* A serial number block runs 00000001–99999999, but the top ~4 million of each
> block are never printed, so the odds use the real printable range of ~96M.

### Odds are for the *whole rule*, aggregated

A pattern's odds are the chance of matching its rule **at all**, counting every
form the rule accepts equally. Take **Ladder Bookend**, whose rule is *"first
digit = last digit, and a run of 4+ consecutive digits somewhere in the middle."*
All of these satisfy that identical rule:

| Serial | Ladder in the middle |
|---|---|
| `02345670` | `234567` — a **6**-long run |
| `02345680` | `23456` — a **5**-long run |
| `02345780` | `2345` — a **4**-long run |

Each counts as **one** hit toward Ladder Bookend's 50,495, so its odds are the
same for all three. **A single pattern's odds can't tell you that one matching
serial is rarer than another** — that gradient lives *inside* the rule, and the
count flattens it.

### The app still ranks their rarity — through the *full match set*

A bill is checked against **every** pattern, and its real rarity and value come
from the **strongest pattern it matches** (and how many it matches), not from any
one pattern's odds. The three serials above show this clearly:

| Serial | Patterns it matches |
|---|---|
| `02345670` | **6 Digit Ladder** + Ladder Bookend |
| `02345680` | Ladder Bookend + **5 Digit Ladder** |
| `02345780` | Ladder Bookend + **4 Digit Ladder** |

`02345670` *is* recognized as the rarest of the three — not via the Ladder
Bookend odds, but because it **also** matches the rarer **6 Digit Ladder**. This
is why the results list can be **sorted by pattern count**: more (and rarer)
matches float the better bills to the top.

### If you want a pattern's odds to reflect a gradient

Split it into stricter variants (e.g. *Ladder Bookend 4 / 5 / 6*). Each variant
is then counted separately, so a 6-long ladder bookend enumerates to far fewer
serials → much longer odds → a higher tier and price. Often this isn't worth the
extra pattern, because the standalone ladder patterns (4/5/6-Digit Ladder)
already capture the length gradient — but the option is there in the Pattern
Manager if a distinction matters to you.

---

## Pattern Manager
_(to be expanded)_ — Browse patterns by library, enable/disable them, edit or
create your own (Wizard and AI Generate), test a serial, move a user pattern
between libraries, and resolve name collisions with a built-in.

## Serial Lookup & Strap Check
_(to be expanded)_ — Type a serial to see which patterns it matches (Serial
Lookup), or check a whole sequential strap's worth of serials at once (Strap
Check).

## Physical-print detections (gas pump, seal shift, plates)
_(to be expanded)_ — Detections that come from the image rather than the serial
digits: gas-pump (misaligned digits), seal shift, and plate/series extraction
with the mule magnifier (press **M**).

## Cropping for listings
_(to be expanded)_ — Generating listing-ready crops, and the standalone Crop
Tool for one-off rare bills.

## Labels
_(to be expanded)_ — The Label Preview & Print tool (Tools → Label Preview,
`Ctrl+Shift+L`) and label profiles.

## Insights report
_(to be expanded)_ — Tools → Insights Report: pattern hit-rate vs. keep-rate
across your scan history.

## Backup & Restore
_(to be expanded)_ — File menu: back up all your settings, patterns, corrections,
and ledger to one portable `.zip`, and restore selectively.
