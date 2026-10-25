# Dollar Detective — User Guide

A practical guide to using Dollar Detective: preparing scans, processing them,
reading results, and managing patterns.

> This guide is a work in progress. Sections marked _(to be expanded)_ are
> stubs — the workflow is described in one line so the outline is complete, and
> the detail will be filled in over time.

## Contents

1. [Preparing a folder: Organize vs. Verify](#preparing-a-folder-organize-vs-verify)
2. [Folder formats the app recognizes](#folder-formats-the-app-recognizes)
3. [The two modes: Manual vs. Monitor](#the-two-modes-manual-vs-monitor)
4. [Processing a folder (Manual mode)](#processing-a-folder-manual-mode) _(to be expanded)_
5. [Scanning straps live (Monitor mode)](#scanning-straps-live-monitor-mode)
6. [Reading the results & overlays](#reading-the-results--overlays) _(to be expanded)_
7. [How odds & rarity work](#how-odds--rarity-work)
8. [Pattern Manager](#pattern-manager)
9. [AI pattern generation (API keys)](#ai-pattern-generation-api-keys)
10. [Pattern prices](#pattern-prices)
11. [Serial Lookup & Strap Check](#serial-lookup--strap-check) _(to be expanded)_
12. [Physical-print detections](#physical-print-detections-gas-pump-seal-shift-plates) _(to be expanded)_
13. [Cropping for listings](#cropping-for-listings) _(to be expanded)_
14. [Labels](#labels)
15. [Insights report](#insights-report) _(to be expanded)_
16. [Backup & Restore](#backup--restore) _(to be expanded)_

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

## The two modes: Manual vs. Monitor

The toggle at the **top-left of the toolbar** switches the app between its two
ways of working. Clicking **Manual** or **Monitor** reconfigures the toolbar for
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

### Monitor mode — "scan straps as I go"

Use this to let the app **watch your scanner's output folder** and file each
strap for you, without pointing it at a folder by hand.

1. Click **Monitor Folder**. The toolbar shows a **Watching:** indicator with the
   folder it's watching and the next strap number.
2. Feed the strap on your scanner (in as many passes as you like — a pause never
   ends the strap).
3. Click **Stop & File Batch**. Everything that arrived is filed as the next
   strap and processed.

> **First-time setup:** the first time you enter Monitor mode, a short **setup
> wizard** offers to walk you through the folders and strap naming it needs.
> You can re-run it any time from **Help → Setup Wizard**, or set the same
> values directly under **Settings → Folders**.

See [Scanning straps live (Monitor mode)](#scanning-straps-live-monitor-mode) for the
optional live-processing option.

### Straps are one shared, numbered sequence

However a strap is created — **File Strap** in Manual mode or **Stop & File
Batch** in Monitor mode — it goes into the **same Straps folder** and takes the
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

| | **Manual mode** | **Monitor mode** |
|---|---|---|
| Bills come from | A folder you already have | Your scanner's output folder, watched live |
| You click | **Process**, then optionally **File Strap** | **Monitor Folder** → **Stop & File Batch** |
| Filing is | Optional (only if you want to keep the run as a strap) | The normal end of a scan |
| Result | The next strap in the sequence | The next strap in the sequence |
| Best for | Re-checking existing scans, folders sent to you | Working straps live at the scanner |

---

## Processing a folder (Manual mode)
_(to be expanded)_ — Point the app at a folder, start processing, and watch the
results fill in. Each bill's serial is read, classified against the enabled
patterns, and flagged fancy or sent to review. When you're done, **File Strap**
files the run as the next strap (see [The two modes](#the-two-modes-manual-vs-scan)).

## Scanning straps live (Monitor mode)

Instead of scanning to a folder and then pointing the app at it, you can let the
app **watch your scanner's output folder** and file each strap for you. Switch to
**Monitor** mode, click **Monitor Folder**, feed the strap (in as many passes as you
like), then click **Stop & File Batch** — everything that arrived is moved into a
new strap under your **Straps** folder and processed. Past straps stay in the
batch dropdown so you can reopen them.

### Live processing (experimental) — depends on how your scanner saves files

There's an experimental **Process live** checkbox that starts classifying scans
*as they arrive* instead of waiting until you click Stop & File Batch. It helps
in two ways, depending on **when your scanner writes the image files**:

- **If the scanner writes each page (or each pass) to the folder as it feeds**,
  live processing works through them *while you keep scanning* — so processing is
  largely done by the time you finish the strap.
- **If the scanner holds the whole strap and writes every file at once at the
  end**, there's nothing to chew on until you finish — but live mode still starts
  the moment the files land (i.e. as soon as you click **Done/Finish** in your
  scanner software), instead of waiting for you to come back and click Stop &
  File Batch. If you tend to step away to stack the strap, that's real time saved.

> **Canon imageFormula R40:** this scanner buffers all pages in the Canon
> software and only writes them to the output folder when you click **Finish** in
> its interface — so every file appears at once. Live processing won't overlap
> with scanning on the R40, but leaving it **on** still lets Dollar Detective
> begin processing the instant you click Finish, rather than idling until you
> click Stop & File Batch. (Behavior varies by scanner and by its "save"
> settings — check yours by watching the output folder while you feed a strap: do
> files trickle in, or appear all at once at the end?)

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

The Pattern Manager is where you browse, turn on/off, create, and edit the
patterns the app looks for. Patterns are grouped into **libraries** (e.g. the
built-in `core` library and your own `user` library).

- **Enable / disable** — tick a pattern to include it in classification, or tick
  a whole library header to toggle the group. Only enabled patterns match bills.
- **Price column** — each pattern's price shows here (from the shared prices
  file, falling back to its built-in value — see **Pattern prices**).
- **Catalog number** — each pattern has a stable catalog number (`C-004`,
  `U-002`, …) used by the prices file and when sharing.
- **Test a serial** — type a serial to see whether a pattern matches and how it
  highlights.
- **Move a user pattern** between libraries, and resolve **name collisions**
  (if two patterns share an internal name, the first wins and the other is
  flagged).

### Creating a pattern — the **New Pattern…** chooser

Click **New Pattern…** and the app asks *how* you want to make it, then takes you
to a single, focused editor for that choice (a **‹ Back** arrow returns to the
chooser). Every editor shares the same top fields — **Name**, **Display Name**,
**Tier**, **Library**, and an optional **Examples** field (sample serials that
match, used by the preview's "Generate Random").

The ways to create a pattern:

- **Simple rule** — a basic "starts with / contains / ends with a value" rule.
- **From a recipe** — pick a shape (ladder, repeater, radar, pairs, palindrome,
  …) and tweak its options. No coding.
- **From a list of numbers** — paste or load a list of serials and the app builds
  a pattern that flags them, with no scripting. Two match types: **whole serial
  is on the list** (short entries are zero-padded to 8 digits) or **serial
  contains a number from the list** (handy for things like years). Very large
  lists are supported — loading a big file shows a summary instead of filling the
  box, and saving a very large list (100k+) warns you first.
- **From a list of dates** — paste specific dates (birthdays, anniversaries,
  etc.), pick how you're typing them (M/D/Y, D/M/Y, or Y/M/D) and which serial
  layouts to match, and the app creates one pattern that catches exactly those
  dates. Invalid dates (e.g. Feb 29 in a non-leap year) are skipped.
- **Start from an existing pattern** — duplicate a pattern you already have (also
  available by right-clicking a pattern, or the **Duplicate…** button). If you
  have an AI key, you can then **Modify with AI** (describe the change) or
  **Edit the Lua** by hand.
- **Describe it in plain English** — have AI write the pattern for you (needs an
  API key — see the next section). Greyed out until a key is set.
- **Write it myself (Lua)** — the advanced option: write the Lua script directly.
  This editor also has **Copy for AI** (copies a ready-made prompt, with your
  current script, to paste into ChatGPT/Claude) and **API Docs**.

A pattern you create is **turned on automatically**, so it starts matching right
away. **Editing** a pattern opens straight into the right editor for it (a list
pattern reopens the list editor, a Lua pattern the Lua editor, and so on).

## AI pattern generation (API keys)

The **Describe it in plain English** option in **New Pattern…** (and the
**Modify with AI** choice when duplicating) lets you describe a pattern in plain
English and have the Lua written for you. It needs an API key from one of two AI
providers — **Anthropic (Claude)** or **OpenAI (GPT)**. You only need **one**.
Each provider charges a small per-use amount; getting a key takes a few minutes.

**Where to enter the key:** **Settings → AI**, in the **API Keys** box. Paste
the key into the Anthropic or OpenAI field and click OK. (The AI Generate tab
also has an **Open AI Settings…** button that takes you straight there.) Keys
are stored locally on your machine in `user_settings.yaml` and are never shared.

**How to get a key — step-by-step videos:**

- **Anthropic (Claude):** <https://www.youtube.com/watch?v=vgncj7MJbVU>
- **OpenAI (GPT):** <https://www.youtube.com/watch?v=OB99E7Y1cMA>

> Anthropic keys start with `sk-ant-…`; OpenAI keys start with `sk-…`. You add
> a little credit to your provider account, create the key, copy it once (the
> provider won't show it again), and paste it into Dollar Detective.

Once a key is saved, the **Describe it in plain English** card becomes available
in New Pattern…; until then it's greyed out with a hint to add a key. The same
links are shown right under the key fields in **Settings → AI**.

## Pattern prices

Each pattern has an estimated price range (shown in the preview panel's **Est.
Price**, result tooltips, and Serial Lookup). These start from a built-in value,
but the group can keep them current with a shared, editable **prices file** —
no need to edit patterns or wait for an app update.

The prices file is a plain CSV keyed by each pattern's **catalog number**
(`C-001`, `U-002`, …), so a price travels with the catalog number even if a
pattern's name changes. The columns are:

`catalog, pattern, display_name, price, price_circ, price_unc`

- **catalog** — *the key.* The pattern's catalog number, exactly as shown in the
  Pattern Manager. This is the only thing that links a row to a pattern.
- **price** — the text that shows in the app (e.g. `$20–$100`, `$50`, or `Ask`).
- **price_circ** — *optional* number: the **circulated** value. It fills the
  Price **column** in the Pattern Manager and is what the list **sorts** by.
  Leave it blank and the app falls back to the low end of the `price` text
  (so `$20–$100` sorts as 20).
- **price_unc** — *optional* number: the **uncirculated** value, shown only
  where you fill it in.
- **pattern / display_name** — only there so the file reads nicely while you
  edit; they are **not** used to match (matching is by `catalog` only).

A row takes effect once it has a catalog number **and** at least one price
field filled in. For example:

```
catalog,pattern,display_name,price,price_circ,price_unc
C-004,RADAR,Radar,$25-$60,25,80
U-002,VAMPIRE,Vampire,$40,40,
C-157,US_DATE_NOTES,Date Note M/D/Y,Ask,,
```

- `C-004` shows "$25-$60", sorts at **25**, and carries an uncirculated value of **80**.
- `U-002` shows "$40" (circulated 40, no uncirculated value).
- `C-157` shows "Ask" — no number, so it has nothing to sort by numerically.

> Older prices files with just `catalog, pattern, display_name, price` still work
> — the two numeric columns are simply treated as blank.

Everything is under **Tools → Pattern Prices**:

- **Open Prices File…** — creates the file (the first time, filled in with the
  current prices for every pattern so you start from a full list — including a
  starter `price_circ` taken from each price) and opens it in your spreadsheet
  app. Edit the price columns, save, and close.
- **Reload Prices** — re-reads the file after you've edited it, so the new
  prices show as you view bills and patterns.
- **Import Prices (CSV)…** — adopt a prices file someone in the group sent you.
- **Export Prices (CSV)…** — save the current prices to a CSV to share with the
  group.

> A pattern with no row in the file (or a blank price) simply keeps its built-in
> price, so you only need rows for the ones you want to change. The file lives in
> your user-data folder (`pattern_prices.csv`) and is never shared automatically
> — the group passes it around with Import/Export.

**Typical group workflow:** one person maintains the prices, edits them with
**Open Prices File…**, then **Export**s the CSV and shares it; everyone else
**Import**s it. Prices update in the app without a new release.

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

The **Label Preview & Print** tool (**Tools → Label Preview**, `Ctrl+Shift+L`)
turns your processed bills into printable labels. It shows one label per bill,
which you can tweak per bill, then save as **PDF** or **Word**.

**Label profile.** A label's layout — its size, fonts, and which fields appear
(catalog number, serial, pattern, position, odds, price, …) — comes from a
label **profile**. Click **Edit…** to change the profile: label size, the base
font, which fields to show, their captions, and per-field fonts.

**Customer vs. Internal versions.** Each field has a **"show on"** setting —
**Both**, **Customer**, or **Internal** — so one profile produces two versions
of every label. For example you might show the catalog number and shelf position
only on the *Internal* version, while the *Customer* version (the one that goes
out with the bill) stays clean. A **Version** picker in Label Preview
(Customer / Internal) lets you see each one.

**Printing both.** The Word export writes the two versions interleaved — the
Customer label on odd pages and the Internal label on even pages — so you can
print each set using Word's built-in **odd/even page** printing. PDF export
follows the version you have selected.

## Insights report
_(to be expanded)_ — Tools → Insights Report: pattern hit-rate vs. keep-rate
across your scan history.

## Backup & Restore
_(to be expanded)_ — File menu: back up all your settings, patterns, corrections,
and ledger to one portable `.zip`, and restore selectively.
