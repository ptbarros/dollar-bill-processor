# The Green Guide — CS Pattern Tracker

**Legend:** ✅ Implemented | 🔲 Todo | ⏸ Deferred | ❌ Image-only

Last updated: 2026-02-22
Implemented: 104 files
Book total: 240 patterns per appendix (CS-10 to CS-2390); CS# verified against ~/projects/tggfsn.ods

---

## Pending Work — DisplayName Audit

**Status:** Audit complete (2026-02-21). 23 DisplayNames corrected across three passes.

### Completed batch 1 (applied 2026-02-21):
- All X0AK → XOAK: cs_30ak, cs_40ak, cs_50ak, cs_60ak, cs_70ak, cs_paired_30ak, cs_random_40ak, cs_double_40ak
- "(Random)" suffix → "CS-Random" prefix: cs_two_pairs, cs_tri_pairs, cs_quad_pairs
- cs_grouped_quad_pairs: "CS-Grouped Quad Pairs" → "CS-Quad Pairs"
- cs_solid: "CS-Solid (CS-80AK)" → "CS-Solid"
- cs_looping_ladder_asc: "CS-Looping Ladder (Ascending)" → "CS-Ascending Looping Ladder"
- cs_looping_ladder_desc: "CS-Looping Ladder (Descending)" → "CS-Descending Looping Ladder"

### Completed batch 2 (applied 2026-02-21):
- cs_full_repeater: "CS-Full Repeater" → "CS-Paired Quad Repeater" (book @CS~1480)
- cs_stand_alone_ladder: "CS-Stand Alone Ladder" → "CS-Stand Alone Mini Ladder" (book CS-1860)

### Completed batch 3 (applied 2026-02-21):
- cs_count_ones: "CS-Count by Ones" → "CS-Count Ones" (book uses no "by")
- cs_count_tens: "CS-Count by Tens" → "CS-Count Tens" (book uses no "by")
- cs_triple_double_double: "CS-Triple-Double-Double" → "CS-Triple Double Double" (spaces, not hyphens)
- cs_triple_triple_pair: "CS-Triple-Triple-Pair" → "CS-Triple Triple Pair" (spaces, not hyphens)
- cs_million_note: "CS-Million Note" → "CS-Million Notes" (book uses plural)
- cs_single_skip_note: "CS-Single Skip Note" → "CS-Single Skip Notes" (book uses plural)

### Verified — no change needed:
- All other implemented patterns confirmed against book

### Naming conventions confirmed from book:
- Book uses "CS-Random XXX" prefix, NOT "CS-XXX (Random)" suffix
- "OAK" = Of A Kind: 2OAK, 3OAK, 4OAK, 5OAK, 6OAK, 7OAK, 8OAK
- "CS-Quad Pairs" (grouped AABBCCDD) — no "Grouped" prefix
- "CS-Ascending Looping Ladder" / "CS-Descending Looping Ladder" (adjective first)
- Bookend names confirmed: CS-Dual Matched Bookend, CS-Tri Matched Bookend, CS-Dual Repeater Bookend, CS-Tri Repeated Bookend, CS-Tri Radar Bookend
- Radar names confirmed: CS-Lucky Seven Radar, CS-Split Six Radar, CS-Oscillating Radar, CS-Bookend Full Radar, CS-Wide Radar, CS-Quad Bookend Radar
- Mini Radar/Repeater names confirmed: CS-Mini 3–7 Radar, CS-Mini 4–7 Repeater
- Source file to search: /tmp/tggfsn.txt
- CS# verified against spreadsheet: ~/projects/tggfsn.ods

---

## Implemented Patterns

| CS# | Book Name | Status | File |
|-----|-----------|--------|------|
| CS-30 | CS-Random Two Pairs | ✅ | cs_two_pairs.lua |
| CS-50 | CS-Random Tri Pairs | ✅ | cs_tri_pairs.lua |
| CS-60 | CS-Quad Pairs | ✅ | cs_grouped_quad_pairs.lua |
| CS-70 | CS-Random Quad Pairs | ✅ | cs_quad_pairs.lua |
| CS-100 | CS-Triple | ✅ | cs_triple.lua |
| CS-110 | CS-3OAK | ✅ | cs_30ak.lua |
| CS-120 | CS-Paired 3OAK | ✅ | cs_paired_30ak.lua |
| CS-130 | CS-Triple Triple Pair | ✅ | cs_triple_triple_pair.lua |
| CS-150 | CS-Double Triples | ✅ | cs_double_triples.lua |
| CS-160 | CS-Random Double Triples | ✅ | cs_random_double_triples.lua |
| CS-170 | CS-Triple Double Double | ✅ | cs_triple_double_double.lua |
| CS-190 | CS-4OAK | ✅ | cs_40ak.lua |
| CS-200 | CS-Quad | ✅ | cs_quad.lua |
| CS-210 | CS-Random 4OAK | ✅ | cs_random_40ak.lua |
| CS-230 | CS-Double Quad | ✅ | cs_double_quad.lua |
| CS-240 | CS-Random Double 4OAK | ✅ | cs_double_40ak.lua |
| CS-250 | CS-Quad in Quad | ✅ | cs_quad_in_quad.lua |
| CS-260 | CS-Quad in Triple | ✅ | cs_quad_in_triple.lua |
| CS-270 | CS-Random Quad in Triple | ✅ | cs_random_quad_in_triple.lua |
| CS-280 | CS-Double Double | ✅ | cs_double_double.lua |
| CS-290 | CS-Triple in Quad | ✅ | cs_triple_in_quad.lua |
| CS-310 | CS-Quad and Pairs | ✅ | cs_quad_and_pairs.lua |
| CS-330 | CS-Pairs in Quad | ✅ | cs_pairs_in_quad.lua |
| CS-360 | CS-5OAK | ✅ | cs_50ak.lua |
| CS-370 | CS-Leading, Center, and Trailing Quints | ✅ | cs_quint.lua |
| CS-380 | CS-Quint in a Pair | ✅ | cs_quint_in_pair.lua |
| CS-400 | CS-Random Quint and Pair | ✅ | cs_random_quint_and_pair.lua |
| CS-410 | CS-Quint in a Triple | ✅ | cs_quint_in_triple.lua |
| CS-420 | CS-Triple in a Quint | ✅ | cs_triple_in_quint.lua |
| CS-430 | CS-6OAK | ✅ | cs_60ak.lua |
| CS-440 | CS-Sextup | ✅ | cs_sextup.lua |
| CS-460 | CS-Pair in a Sextup | ✅ | cs_pair_in_sextup.lua |
| CS-470 | CS-Random Pair in a Sextup | ✅ | cs_random_pair_in_sextup.lua |
| CS-480 | CS-Seven | ✅ | cs_seven.lua |
| CS-490 | CS-7OAK | ✅ | cs_70ak.lua |
| CS-500 | CS-Solid | ✅ | cs_solid.lua |
| CS-710 | CS-Double Year Note | ✅ | double_year.lua |
| CS-810 | CS-Count Ones | ✅ | cs_count_ones.lua |
| CS-820 | CS-Count Tens | ✅ | cs_count_tens.lua |
| CS-900 | CS-True Binary | ✅ | cs_true_binary.lua |
| CS-910 | CS-Binary | ✅ | cs_binary.lua |
| CS-920 | CS-True Double Quad Binary | ✅ | cs_true_double_quad_binary.lua |
| CS-930 | CS-Random Double Quad Binary | ✅ | cs_random_double_quad_binary.lua |
| CS-940 | CS-Trinary | ✅ | cs_trinary.lua |
| CS-950 | CS-Single Bookend | ✅ | cs_single_bookend.lua |
| CS-960 | CS-Dual Matched Bookend | ✅ | cs_dual_bookend.lua |
| CS-980 | CS-Dual Repeater Bookend | ✅ | cs_dual_repeater_bookend.lua |
| CS-990 | CS-Tri Matched Bookend | ✅ | cs_tri_bookend.lua |
| CS-1000 | CS-Tri Repeated Bookend | ✅ | cs_tri_repeated_bookend.lua |
| CS-1010 | CS-Tri Radar Bookend | ✅ | cs_tri_radar_bookend.lua |
| CS-1040 | CS-True Binary Flipper | ✅ | cs_true_binary_flipper.lua |
| CS-1050 | CS-Binary Flipper | ✅ | cs_binary_flipper.lua |
| CS-1060 | CS-Trinary Flipper | ✅ | cs_trinary_flipper.lua |
| CS-1070 | CS-Quad Flipper | ✅ | cs_quad_flipper.lua |
| CS-1090 | CS-Rotator | ✅ | cs_rotator.lua |
| CS-1160 | CS-Tetradic | ✅ | cs_tetradic.lua |
| CS-1170 | CS-Ascending Ladder | ✅ | cs_ascending_ladder.lua |
| CS-1180 | CS-Descending Ladder | ✅ | cs_descending_ladder.lua |
| CS-1190 | CS-Ascending Looping Ladder | ✅ | cs_looping_ladder_asc.lua |
| CS-1200 | CS-Descending Looping Ladder | ✅ | cs_looping_ladder_desc.lua |
| CS-1210 | CS-Scattered Ladder | ✅ | cs_scattered_ladder.lua |
| CS-1230 | CS-Ascending Broken Ladder | ✅ | cs_ascending_broken_ladder.lua |
| CS-1240 | CS-Descending Broken Ladder | ✅ | cs_descending_broken_ladder.lua |
| CS-1260 | CS-Super Radar | ✅ | cs_super_radar.lua |
| CS-1270 | CS-Full Radar | ✅ | cs_full_radar.lua |
| CS-1280 | CS-Bookend Full Radar | ✅ | cs_bookend_full_radar.lua |
| CS-1290 | CS-Wide Radar | ✅ | cs_wide_radar.lua |
| CS-1300 | CS-Split Six Radar | ✅ | cs_split_six_radar.lua |
| CS-1310 | CS-Quad Bookend Radar | ✅ | cs_quad_bookend_radar.lua |
| CS-1320 | CS-Pinpoint Radar | ✅ | cs_pinpoint_radar.lua |
| CS-1330 | CS-Oscillating Radar | ✅ | cs_oscillating_radar.lua |
| CS-1350 | CS-Lucky Seven Radar | ✅ | cs_lucky_seven_radar.lua |
| CS-1370 | CS-Mini 3 Radar | ✅ | cs_mini_3_radar.lua |
| CS-1380 | CS-Mini 4 Radar | ✅ | cs_mini_4_radar.lua |
| CS-1390 | CS-Mini 5 Radar | ✅ | cs_mini_5_radar.lua |
| CS-1400 | CS-Mini 6 Radar | ✅ | cs_mini_6_radar.lua |
| CS-1410 | CS-Mini 7 Radar | ✅ | cs_mini_7_radar.lua |
| CS-1480 | CS-Paired Quad Repeater | ✅ | cs_full_repeater.lua |
| CS-1520 | CS-Radar Repeater | ✅ | cs_radar_repeater.lua |
| CS-1530 | CS-Super Repeater | ✅ | cs_super_repeater.lua |
| CS-1550 | CS-Mini 4 Repeater | ✅ | cs_mini_4_repeater.lua |
| CS-1560 | CS-Mini 5 Repeater | ✅ | cs_mini_5_repeater.lua |
| CS-1570 | CS-Mini 6 Repeater | ✅ | cs_mini_6_repeater.lua |
| CS-1580 | CS-Mini 7 Repeater | ✅ | cs_mini_7_repeater.lua |
| CS-1590 | CS-Single Skip Notes | ✅ | cs_single_skip_note.lua |
| CS-1610 | CS-Skip Count Up Note | ✅ | cs_skip_count_up.lua |
| CS-1620 | CS-Skip Count Down Note | ✅ | cs_skip_count_down.lua |
| CS-1650 | CS-Stand Alone Singles | ✅ | cs_stand_alone_single.lua |
| CS-1660 | CS-Stand Alone Pair | ✅ | cs_stand_alone_pair.lua |
| CS-1670 | CS-Stand Alone Triple | ✅ | cs_stand_alone_triple.lua |
| CS-1680 | CS-Stand Alone Quad | ✅ | cs_stand_alone_quad.lua |
| CS-1690 | CS-Stand Alone Quint | ✅ | cs_stand_alone_quint.lua |
| CS-1710 | CS-Stand Alone Double Repeater | ✅ | cs_stand_alone_double_repeater.lua |
| CS-1720 | CS-Stand Alone Tri Repeater | ✅ | cs_stand_alone_tri_repeater.lua |
| CS-1730 | CS-Stand Alone Mini 3 Radar | ✅ | cs_stand_alone_mini_3_radar.lua |
| CS-1740 | CS-Stand Alone Mini 4 Radar | ✅ | cs_stand_alone_mini_4_radar.lua |
| CS-1810 | CS-Stand Alone Year | ✅ | cs_stand_alone_year.lua |
| CS-1860 | CS-Stand Alone Mini Ladder | ✅ | cs_stand_alone_ladder.lua |
| CS-1940 | CS-Leading Zeros | ✅ | cs_leading_zeros.lua |
| CS-1950 | CS-Centered Zeros | ✅ | cs_centered_zeros.lua |
| CS-1960 | CS-Trailing Zeros | ✅ | cs_trailing_zeros.lua |
| CS-1970 | CS-High Notes | ✅ | cs_high_notes.lua |
| CS-1980 | CS-Low Notes | ✅ | cs_low_notes.lua |
| CS-1990 | CS-Million Notes | ✅ | cs_million_note.lua |

---

## Todo — Future Batches

Implementable with pure serial-number Lua logic (no metadata required).

| CS# | Book Name | Notes |
|-----|-----------|-------|
| CS-10 | CS-2OAKs | Any two matching digits anywhere in serial |
| CS-20 | CS-Two Pairs | Verify definition vs CS-30 (Random Two Pairs) |
| CS-40 | CS-Tri Pairs | Consecutive grouped pairs AABBCC; previously attempted, removed — needs book re-read |
| CS-80 | CS-Pairs in Pairs | Needs book definition |
| CS-90 | CS-Random Pairs in Pairs | Needs book definition |
| CS-140 | CS-Random Triple Triple Pair | Scattered variant of CS-130 |
| CS-180 | CS-Random Triple Double Double | Scattered variant of CS-170 |
| CS-220 | CS-Leading, Center & Trailing Quads | Positional variants of CS-Quad; distinct from CS-210 (Random 4OAK) |
| CS-300 | CS-Random Triple in Quad | Scattered triple within quad context |
| CS-320 | CS-Random Quad and Pairs | Scattered variant of CS-310 |
| CS-340 | CS-Quads and Triples | Quad + triple combination |
| CS-350 | CS-Repeating Doubles | Needs further book research |
| CS-390 | CS-Pair in a Quint | Pair somewhere within quint context |
| CS-450 | CS-Pair and a Sextup | Pair + sextup combination |
| CS-830 | CS-Count Hundreds | Counting pattern variant |
| CS-840 | CS-Count One Quads | Counting pattern variant |
| CS-850 | CS-Count Ten Quads | Counting pattern variant |
| CS-860 | CS-Count Hundred Quads | Counting pattern variant |
| CS-870 | CS-Count Thousand Quad | Counting pattern variant |
| CS-880 | CS-Double Quads Count Note | Counting pattern variant |
| CS-890 | CS-Split Count | Counting pattern variant |
| CS-970 | CS-Dual Radar Bookend | Palindrome bookend at both ends; distinct from CS-960 (Dual Matched Bookend) |
| CS-1030 | CS-Unary Flipper | All 8 digits the same flip-valid digit |
| CS-1080 | CS-Quinary Flipper | All 5 flip digits present {0,1,6,8,9} |
| CS-1100 | CS-Unary Rotator | Needs rotator research |
| CS-1110 | CS-True Binary Rotator | May overlap CS-1040 (True Binary Flipper) — deferred pending clarification |
| CS-1120 | CS-Binary Rotator | Needs rotator research |
| CS-1130 | CS-Trinary Rotator | Needs rotator research |
| CS-1140 | CS-Quad Rotator | Needs rotator research |
| CS-1150 | CS-Quinary Rotator | Needs rotator research |
| CS-1220 | CS-Broken Ladder | Superset of CS-1230/CS-1240; skip until subtypes are stable |
| CS-1250 | CS-Buildable Ladder | Complex definition; needs more book research |
| CS-1340 | CS-Shotgun Radar | Distinct from CS-Mini 3 Radar (CS-1370); needs book definition |
| CS-1420 | CS-Ascending Laddered Radar | Ascending ladder embedded in radar structure |
| CS-1430 | CS-Descending Laddered Radar | Descending ladder embedded in radar structure |
| CS-1440 | CS-Repeater | General repeater (ABCDABCD); verify vs CS-1480 (Paired Quad Repeater) |
| CS-1450 | CS-Single Bookend Repeater | Repeater with single bookend |
| CS-1460 | CS-Dual Bookend Repeater | Repeater with dual bookend |
| CS-1470 | CS-Tri Bookend Repeater | Repeater with tri bookend |
| CS-1490 | CS-Random Quad Repeater | Scattered quad repeater variant |
| CS-1500 | CS-Triple Repeater | Three-segment repeater |
| CS-1510 | CS-Six Repeater in a Pair | Sextup repeater in pair context |
| CS-1600 | CS-Double Skip Notes | Same as CS-Super Repeater (ABABABAB) — already covered by CS-1530 |
| CS-1630 | CS-True Binary Skip Note | Binary variant of skip note |
| CS-1700 | CS-Stand Alone Sextup | 6-digit run surrounded by zeros |
| CS-1750 | CS-Stand Alone Mini 5 Radar | Mini 5-digit palindrome surrounded by zeros |
| CS-1760 | CS-Stand Alone Tri Radar | Tri radar surrounded by zeros |
| CS-1770 | CS-Stand Alone Quad Radar | Quad radar surrounded by zeros |
| CS-1870 | CS-Stand Alone Mini Up Ladder 4 | 4-digit ascending ladder surrounded by zeros |
| CS-1880 | CS-Stand Alone Mini Down Ladder 4 | 4-digit descending ladder surrounded by zeros |
| CS-1890 | CS-Stand Alone Mini Up Ladder 5 | 5-digit ascending ladder surrounded by zeros |
| CS-1900 | CS-Stand Alone Mini Down Ladder 5 | 5-digit descending ladder surrounded by zeros |
| CS-1910 | CS-Stand Alone Mini Up Ladder 6 | 6-digit ascending ladder surrounded by zeros |
| CS-1920 | CS-Stand Alone Mini Down Ladder 6 | 6-digit descending ladder surrounded by zeros |
| CS-1930 | CS-Random Zeros | Zeros scattered throughout serial |
| CS-2280 | CS-Zip Codes | Serial matches a US zip code format |
| CS-2290 | CS-Prime Numbers | Serial number is mathematically prime |
| CS-2300 | CS-Phone Notes | Serial matches phone number format |
| CS-2380 | CS-Odds and Evens | Serial digits alternate or group odd/even |
| CS-2390 | CS-Sequential Numbers | Serial forms a sequential numeric run |

---

## Deferred — Date/Calendar Patterns

These require calendar validation (valid date math) or `ctx.metadata.series_year`. Save for a dedicated date batch.

| CS# | Book Name | Reason Deferred |
|-----|-----------|-----------------|
| CS-1780 | CS-Stand Alone Date | Section header; encompasses CS-1790 and CS-1800 |
| CS-1790 | CS-Stand Alone Date US & INTL | mm/dd block surrounded by zeros; needs valid date validation |
| CS-1800 | CS-Stand Alone Date E.U. | dd/mm block surrounded by zeros; needs valid date validation |
| CS-1820 | CS-Stand Alone Date Year | Date + 2-digit year block; needs date+year validation |
| CS-1830 | CS-Stand Alone US Date Year | Needs calendar logic to verify year is a real US date year |
| CS-1840 | CS-Stand Alone EU Date Year | EU date + year; needs date validation |
| CS-1850 | CS-Stand Alone INTL Date Year | INTL date + year; needs date validation |
| CS-540 | CS-US Birthday Note | Requires series year match to a date |
| CS-550 | CS-US Leap Year History Note | Calendar date + leap year math |
| CS-560 | CS-US Future Date Note | Date must be in the future relative to series year |
| CS-570 | CS-EU Birthday Note | EU date format calendar match |
| CS-580 | CS-EU Leap Year Birthday Note | EU date + leap year |
| CS-590 | CS-EU History Note | EU date calendar match |
| CS-600 | CS-EU Leap Year History Note | EU date + leap year |
| CS-610 | CS-EU Future Date Note | EU date in future |
| CS-620 | CS-INTL Birthday Note | INTL date format calendar match |
| CS-630 | CS-INTL Leap Year Birthday Note | INTL date + leap year |
| CS-640 | CS-INTL History Note | INTL date calendar match |
| CS-650 | CS-INTL Future Date Note | INTL date in future |
| CS-660 | CS-INTL Leap Year History Note | INTL date + leap year |
| CS-670 | CS-True Year Note | series_year in serial |
| CS-680 | CS-Numbered Year Note | Year appears in serial as a numbered reference |
| CS-690 | CS-Random Year Note | Year scattered in serial |
| CS-700 | CS-Year Notes | Section header for year-based patterns |
| CS-720 | CS-Triple Year Note | Year appears three times |
| CS-730 | CS-Quad Year Note | Year appears four times |
| CS-740 | CS-Quint Year Note | Year appears five times |
| CS-750 | CS-Day Notes | Section header; serial forms a day pattern |
| CS-760 | CS-True Day Note | Serial digits form exact day |
| CS-770 | CS-US & INTL True Day Notes | US/INTL format day match |
| CS-780 | CS-EU True Day Notes | EU format day match |
| CS-790 | CS-Numbered Day Note | Day appears as numbered reference |
| CS-800 | CS-Random Day Month Note | Random day/month combination |

---

## Image-Only — Not Implementable via Serial Analysis

These require visual inspection of the physical note (printing errors, stamps, etc.).

| CS# | Book Name | Reason |
|-----|-----------|--------|
| CS-2000 | CS-Web Notes | Web press identification — visual |
| CS-2010 | CS-Gas Pumps | Vertical misalignment — handled separately via image analysis |
| CS-2020 | CS-Over Inked | Printing error — visual |
| CS-2030 | CS-Off Center Printing | Printing error — visual |
| CS-2040 | CS-Ink Smears | Printing error — visual |
| CS-2050 | CS-Ink Splatter | Printing error — visual |
| CS-2060 | CS-Ink Transfer | Printing error — visual |
| CS-2070 | CS-Off Center Cuts | Printing error — visual |
| CS-2080 | CS-Wet Sheet Transfer | Printing error — visual |
| CS-2090 | CS-Insufficient Ink | Printing error — visual |
| CS-2100 | CS-Wrong Back Plate & CS-Wrong Front Plate | Plate error — visual |
| CS-2110 | CS-Fold Over or CS-Gutter Fold | Printing error — visual |
| CS-2120 | CS-Doubling | Printing error — visual |
| CS-2130 | CS-Inverted Back | Printing error — visual |
| CS-2140 | CS-Blank Reverse & CS-Blank Obverse | Printing error — visual |
| CS-2150 | CS-Obstructed Printings | Printing error — visual |
| CS-2160 | CS-Offset Transfer | Printing error — visual |
| CS-2170 | CS-Reverse Overprints | Printing error — visual |
| CS-2180 | CS-Misaligned Overprints | Printing error — visual |
| CS-2190 | CS-Missing Overprint | Printing error — visual |
| CS-2200 | CS-Missing Front Printing | Printing error — visual |
| CS-2210 | CS-Mismatched Serial Numbers | Serial mismatch — visual comparison |
| CS-2220 | CS-Zegers/Winograd Project | Special project — visual |
| CS-2230 | CS-106 Error | Production error — visual |
| CS-2240 | CS-7273 Error | Production error — visual |
| CS-2250 | CS-129 Error | Production error — visual |
| CS-2260 | CS-295 Error | Production error — visual |
| CS-2270 | CS-905 Error | Production error — visual |
| CS-2310 | CS-Personalized Stamps | Stamps — visual |
| CS-2320 | CS-Bank Stamps | Stamps — visual |
| CS-2330 | CS-Where's George Stamps | Stamps — visual |
| CS-2340 | CS-Political Stamps | Stamps — visual |
| CS-2350 | CS-Signatures (Celebrity) | Signatures — visual |
| CS-2360 | CS-Bank Dye Packs | Dye pack evidence — visual |
| CS-2370 | CS-Star Notes | Star replacement notes — detected separately |

---

## Notes

- CS-100 = CS-Triple (consecutive 3-in-a-row); CS-110 = CS-3OAK (scattered) — these are different patterns; previous TRACKING.md had the row labels swapped but the files (BookRef CS-100/CS-110) were already correct
- CS-190 = CS-4OAK (scattered); CS-200 = CS-Quad (consecutive grouped); CS-210 = CS-Random 4OAK — files corrected 2026-02-22
- CS-220 = CS-Leading, Center & Trailing Quads — this is NOT the same as CS-Random 4OAK; it is a positional pattern (quad at start, center, or end)
- CS-370 = CS-Leading, Center, and Trailing Quints (implemented as cs_quint.lua — any consecutive 5-of-a-kind)
- CS-430 = CS-6OAK; CS-440 = CS-Sextup (consecutive grouped) — separate patterns
- CS-500 = CS-Solid (CS-8OAK, all 8 digits the same)
- CS-710 (double_year.lua): verify this is CS-710 vs CS-720 — book may number differently
- CS-1060 = CS-Trinary Flipper (3 distinct flip digits); CS-1070 = CS-Quad Flipper (4 distinct flip digits) — confirmed CS# from book appendix; previously listed as "(no CS#)"
- CS-1090 = CS-Rotator (general); CS-1100–CS-1150 are specific rotator subtypes (Unary, Binary, Trinary, Quad, Quinary)
- CS-1260 = CS-Super Radar (ABBBBBBA structure) — confirmed CS# from book appendix
- CS-1340 = CS-Shotgun Radar (not yet implemented); previously mislabeled in TRACKING.md as CS-Mini 3 Radar
- CS-1370 = CS-Mini 3 Radar (cs_mini_3_radar.lua) — BookRef corrected from CS-1340 to CS-1370
- CS-1860 = CS-Stand Alone Mini Ladder (the general cs_stand_alone_ladder.lua pattern); CS-1870–CS-1920 are specific directional/length subtypes not yet implemented
- CS-1890 (CS-Stand Alone Mini Up Ladder 5) remains in Todo — it is a specific subtype; the note that it was "covered by CS-1880" was incorrect since CS-1880 is CS-Stand Alone Mini Down Ladder 4
- Batch 3 added Stand Alone patterns (CS-1650–CS-1740 range)
- Batch 4 added nested/combined group patterns (CS-120, CS-250–CS-470 range) and CS-1860, CS-1990
- Batch 5 added bookend variants (CS-980/1000/1010), binary flipper (CS-1050), stand-alone triple/quad/mini-3-radar (CS-1670/1680/1730), centered zeros (CS-1950), quad pairs (CS-60), random double 40AK (CS-240), random quad in triple (CS-270)
- Batch 6 added: CS-Trinary Flipper (CS-1060), CS-Quad Flipper (CS-1070), CS-True Double Quad Binary (CS-920), CS-Random Double Quad Binary (CS-930), CS-Scattered Ladder (CS-1210), CS-Ascending Broken Ladder (CS-1230), CS-Descending Broken Ladder (CS-1240), CS-Pinpoint Radar (CS-1320), CS-Pairs in Quad (CS-330), CS-Stand Alone Year (CS-1810)
- CS-150 = CS-Double Triples (grouped consecutive triples); CS-160 = CS-Random Double Triples (scattered) — both implemented; CS-150 BookRef was missing, added 2026-02-22
- CS-1480 = CS-Paired Quad Repeater (ABCDABCD); file is cs_full_repeater.lua
- CS-1600 (Double Skip Note) = CS-Super Repeater (already implemented as CS-1530)
- CS-1610/1620 are Skip Count notes, NOT ladder variants
- CS-2280 = CS-Zip Codes (previously mislabeled "CS-Mismatched Serial" in TRACKING.md)
- CS-2290 = CS-Prime Numbers (previously mislabeled "CS-Radar Serial Letters" in TRACKING.md)
- CS# verified 2026-02-22 against ~/projects/tggfsn.ods spreadsheet (book appendix page numbers and CS# confirmed accurate)
