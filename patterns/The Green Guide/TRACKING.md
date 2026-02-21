# The Green Guide — CS Pattern Tracker

**Legend:** ✅ Implemented | 🔲 Todo | ⏸ Deferred | ❌ Image-only

Last updated: 2026-02-21
Implemented: 104 files (94 previous + 10 batch 6)
Book total: ~134 CS-numbered patterns (CS-30 to CS-2350)

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
- cs_stand_alone_ladder: "CS-Stand Alone Ladder" → "CS-Stand Alone Mini Ladder" (book CS-1880)

### Completed batch 3 (applied 2026-02-21):
- cs_count_ones: "CS-Count by Ones" → "CS-Count Ones" (book uses no "by")
- cs_count_tens: "CS-Count by Tens" → "CS-Count Tens" (book uses no "by")
- cs_triple_double_double: "CS-Triple-Double-Double" → "CS-Triple Double Double" (spaces, not hyphens)
- cs_triple_triple_pair: "CS-Triple-Triple-Pair" → "CS-Triple Triple Pair" (spaces, not hyphens)
- cs_million_note: "CS-Million Note" → "CS-Million Notes" (book uses plural)
- cs_single_skip_note: "CS-Single Skip Note" → "CS-Single Skip Notes" (book uses plural)

### Verified — no change needed:
- All other 81 implemented patterns confirmed against book

### Naming conventions confirmed from book:
- Book uses "CS-Random XXX" prefix, NOT "CS-XXX (Random)" suffix
- "OAK" = Of A Kind: 2OAK, 3OAK, 4OAK, 5OAK, 6OAK, 7OAK, 8OAK
- "CS-Quad Pairs" (grouped AABBCCDD) — no "Grouped" prefix
- "CS-Ascending Looping Ladder" / "CS-Descending Looping Ladder" (adjective first)
- Bookend names confirmed: CS-Dual Matched Bookend, CS-Tri Matched Bookend, CS-Dual Repeater Bookend, CS-Tri Repeated Bookend, CS-Tri Radar Bookend
- Radar names confirmed: CS-Lucky Seven Radar, CS-Split Six Radar, CS-Oscillating Radar, CS-Bookend Full Radar, CS-Wide Radar, CS-Quad Bookend Radar
- Mini Radar/Repeater names confirmed: CS-Mini 3–7 Radar, CS-Mini 4–7 Repeater
- Source file to search: /tmp/tggfsn.txt

---

## Implemented Patterns

| CS# | Book Name | Status | File |
|-----|-----------|--------|------|
| CS-30 | CS-Random Two Pairs | ✅ | cs_two_pairs.lua |
| CS-50 | CS-Random Tri Pairs | ✅ | cs_tri_pairs.lua |
| CS-60 (approx) | CS-Quad Pairs | ✅ | cs_grouped_quad_pairs.lua |
| CS-70 | CS-Random Quad Pairs | ✅ | cs_quad_pairs.lua |
| CS-100 | CS-3OAK | ✅ | cs_30ak.lua |
| CS-110 | CS-Triple | ✅ | cs_triple.lua |
| CS-120 | CS-Paired 3OAK | ✅ | cs_paired_30ak.lua |
| CS-130 | CS-Triple Triple Pair | ✅ | cs_triple_triple_pair.lua |
| CS-160 | CS-Random Double Triples | ✅ | cs_random_double_triples.lua |
| CS-170 | CS-Triple Double Double | ✅ | cs_triple_double_double.lua |
| CS-200 | CS-Quad (grouped) | ✅ | cs_quad.lua |
| CS-210 | CS-4OAK | ✅ | cs_40ak.lua |
| CS-220 | CS-Random 4OAK | ✅ | cs_random_40ak.lua |
| CS-230 | CS-Double Quad | ✅ | cs_double_quad.lua |
| CS-240 (approx) | CS-Random Double 4OAK | ✅ | cs_double_40ak.lua |
| CS-250 | CS-Quad in Quad | ✅ | cs_quad_in_quad.lua |
| CS-260 | CS-Quad in Triple | ✅ | cs_quad_in_triple.lua |
| CS-270 | CS-Random Quad in Triple | ✅ | cs_random_quad_in_triple.lua |
| CS-280 | CS-Double Double | ✅ | cs_double_double.lua |
| CS-290 | CS-Triple in Quad | ✅ | cs_triple_in_quad.lua |
| CS-310 | CS-Quad and Pairs | ✅ | cs_quad_and_pairs.lua |
| CS-330 | CS-Pairs in Quad | ✅ | cs_pairs_in_quad.lua |
| CS-360 | CS-5OAK | ✅ | cs_50ak.lua |
| CS-370 | CS-Quint | ✅ | cs_quint.lua |
| CS-380 | CS-Quint in a Pair | ✅ | cs_quint_in_pair.lua |
| CS-400 | CS-Random Quint and Pair | ✅ | cs_random_quint_and_pair.lua |
| CS-410 | CS-Quint in a Triple | ✅ | cs_quint_in_triple.lua |
| CS-420 | CS-Triple in a Quint | ✅ | cs_triple_in_quint.lua |
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
| CS-1090 | CS-Rotator | ✅ | cs_rotator.lua |
| CS-1160 | CS-Tetradic | ✅ | cs_tetradic.lua |
| CS-1170 | CS-Ascending Ladder | ✅ | cs_ascending_ladder.lua |
| CS-1180 | CS-Descending Ladder | ✅ | cs_descending_ladder.lua |
| CS-1190 | CS-Ascending Looping Ladder | ✅ | cs_looping_ladder_asc.lua |
| CS-1200 | CS-Descending Looping Ladder | ✅ | cs_looping_ladder_desc.lua |
| CS-1210 | CS-Scattered Ladder | ✅ | cs_scattered_ladder.lua |
| CS-1230 | CS-Ascending Broken Ladder | ✅ | cs_ascending_broken_ladder.lua |
| CS-1240 | CS-Descending Broken Ladder | ✅ | cs_descending_broken_ladder.lua |
| CS-1270 | CS-Full Radar | ✅ | cs_full_radar.lua |
| CS-1280 | CS-Bookend Full Radar | ✅ | cs_bookend_full_radar.lua |
| CS-1290 | CS-Wide Radar | ✅ | cs_wide_radar.lua |
| CS-1300 | CS-Split Six Radar | ✅ | cs_split_six_radar.lua |
| CS-1310 | CS-Quad Bookend Radar | ✅ | cs_quad_bookend_radar.lua |
| CS-1320 | CS-Pinpoint Radar | ✅ | cs_pinpoint_radar.lua |
| CS-1330 | CS-Oscillating Radar | ✅ | cs_oscillating_radar.lua |
| CS-1340 | CS-Mini 3 Radar / Mini 3 Repeater | ✅ | cs_mini_3_radar.lua |
| CS-1350 | CS-Lucky Seven Radar | ✅ | cs_lucky_seven_radar.lua |
| CS-1380 | CS-Mini 4 Radar | ✅ | cs_mini_4_radar.lua |
| CS-1390 | CS-Mini 5 Radar | ✅ | cs_mini_5_radar.lua |
| CS-1400 | CS-Mini 6 Radar | ✅ | cs_mini_6_radar.lua |
| CS-1410 | CS-Mini 7 Radar | ✅ | cs_mini_7_radar.lua |
| CS-1480 | CS-Paired Quad Repeater | ✅ | cs_full_repeater.lua |
| CS-1520 | CS-Radar Repeater | ✅ | cs_radar_repeater.lua |
| CS-1530 | CS-Super Repeater | ✅ | cs_super_repeater.lua |
| CS-1550 | CS-Mini 4 Repeater (ABAB) | ✅ | cs_mini_4_repeater.lua |
| CS-1560 | CS-Mini 5 Repeater (ABxAB) | ✅ | cs_mini_5_repeater.lua |
| CS-1570 | CS-Mini 6 Repeater (ABCABC) | ✅ | cs_mini_6_repeater.lua |
| CS-1580 | CS-Mini 7 Repeater (ABCxABC) | ✅ | cs_mini_7_repeater.lua |
| CS-1590 | CS-Single Skip Note | ✅ | cs_single_skip_note.lua |
| CS-1610 | CS-Skip Count Up Note | ✅ | cs_skip_count_up.lua |
| CS-1620 | CS-Skip Count Down Note | ✅ | cs_skip_count_down.lua |
| CS-1650 | CS-Stand Alone Single | ✅ | cs_stand_alone_single.lua |
| CS-1660 | CS-Stand Alone Pair | ✅ | cs_stand_alone_pair.lua |
| CS-1670 | CS-Stand Alone Triple | ✅ | cs_stand_alone_triple.lua |
| CS-1680 | CS-Stand Alone Quad | ✅ | cs_stand_alone_quad.lua |
| CS-1690 | CS-Stand Alone Quint | ✅ | cs_stand_alone_quint.lua |
| CS-1710 | CS-Stand Alone Double Repeater | ✅ | cs_stand_alone_double_repeater.lua |
| CS-1720 | CS-Stand Alone Tri Repeater | ✅ | cs_stand_alone_tri_repeater.lua |
| CS-1730 | CS-Stand Alone Mini 3 Radar | ✅ | cs_stand_alone_mini_3_radar.lua |
| CS-1740 | CS-Stand Alone Mini 4 Radar | ✅ | cs_stand_alone_mini_4_radar.lua |
| CS-1880 | CS-Stand Alone Mini Ladder | ✅ | cs_stand_alone_ladder.lua |
| CS-1940 | CS-Leading Zeros | ✅ | cs_leading_zeros.lua |
| CS-1950 | CS-Centered Zeros | ✅ | cs_centered_zeros.lua |
| CS-1960 | CS-Trailing Zeros | ✅ | cs_trailing_zeros.lua |
| CS-1970 | CS-High Notes | ✅ | cs_high_notes.lua |
| CS-1980 | CS-Low Notes | ✅ | cs_low_notes.lua |
| CS-1810 | CS-Stand Alone Year | ✅ | cs_stand_alone_year.lua |
| CS-1990 | CS-Million Note | ✅ | cs_million_note.lua |
| (no CS#) | CS-Double Triples | ✅ | cs_double_triples.lua |
| (no CS#) | CS-6OAK | ✅ | cs_60ak.lua |
| (no CS#) | CS-Super Radar | ✅ | cs_super_radar.lua |
| (no CS#) | CS-Trinary Flipper | ✅ | cs_trinary_flipper.lua |
| (no CS#) | CS-Quad Flipper | ✅ | cs_quad_flipper.lua |

---

## Todo — Future Batches

Implementable with pure serial-number Lua logic.

| CS# | Book Name | Notes |
|-----|-----------|-------|
| CS-350 | CS-Repeating Doubles | Needs further book research |
| CS-960 | (Dual Radar Bookend) | 23xxxx32 mirror at each end — note: overlaps with CS-960 |
| CS-1110 | CS-True Binary Rotator | Rotator subtype; may overlap CS-1040 (True Binary Flipper) — deferred pending clarification |
| CS-1130 | CS-Trinary Rotator | Rotator subtype; needs rotator research |
| CS-1140 | CS-Quad Rotator | Rotator subtype; needs rotator research |
| CS-1220 | CS-Broken Ladder | Superset of CS-1230/CS-1240 (ascending + descending); skip until subtypes are stable |
| CS-1250 | CS-Buildable Ladder | Complex definition; needs more book research |
| CS-1600 | CS-Double Skip Note | Same as CS-Super Repeater (ABABABAB) — already covered by CS-1530 |
| CS-2280 | CS-Mismatched Serial | Unusual serial letter/number combination |
| CS-2290 | CS-Radar Serial Letters | Palindrome in prefix/suffix letters |

---

## Deferred — Date/Metadata Patterns

These require `ctx.metadata.series_year`, plate numbers, or complex calendar math. Save for a dedicated metadata batch.

| CS# | Book Name | Reason Deferred |
|-----|-----------|-----------------|
| CS-1830 | CS-Stand Alone U.S. Date Year | Needs calendar logic to verify year is a real US date year; deferred to metadata batch |
| CS-540 | CS-Birthday Note | Requires series year match to a date |
| CS-550 | CS-Anniversary Note | Calendar date math |
| CS-560 | CS-Calendar Note | Date-based matching |
| CS-610 | CS-Year Note | series_year in serial |
| CS-620 | CS-Double Year Note | Year appears twice |
| CS-630 | CS-Pair Year Note | Year as pair |
| CS-650 | CS-Sequential Year | Year digits sequential in serial |
| CS-660 | CS-Reverse Year | Year digits reversed in serial |
| CS-670 | CS-Year Palindrome | Year forms palindrome |
| CS-690 | CS-Decade Note | Serial matches a decade |
| CS-700 | CS-Century Note | Serial matches a century year |
| CS-720 | CS-Double Year (variant) | Similar to CS-710 — verify overlap |
| CS-730 | CS-Near Year | Serial close to series year |
| CS-740 | CS-Year Ladder | Year digits form a ladder |
| CS-750 | CS-Year Bookend | Year bookends serial |
| CS-760 | CS-Year Repeater | Year repeats in serial |
| CS-780 | CS-Year Radar | Year forms palindrome in serial |
| CS-790 | CS-Year Serial Match | Full serial matches year |
| CS-800 | CS-Plate Match | Serial digits match plate number |
| CS-830 | CS-Birthday Ladder | Birthday date forms ladder |
| CS-840 | CS-Birthday Palindrome | Birthday forms palindrome |
| CS-850 | CS-Holiday Note | Serial matches a holiday date |
| CS-860 | CS-Tax Day Note | Serial = 04151040 or similar |
| CS-870 | CS-Pi Note | Serial approximates π |
| CS-880 | CS-Fibonacci Note | Serial digits follow Fibonacci |
| CS-890 | CS-Prime Note | Serial is a prime number |

---

## Image-Only — Not Implementable via Serial Analysis

These require visual inspection of the physical note (printing errors, signatures, stamps).

| CS# | Book Name | Reason |
|-----|-----------|--------|
| CS-2020 | CS-Ink Smear | Printing error — visual |
| CS-2030 | CS-Digit Offset | Printing error — visual |
| CS-2080 | CS-Double Print | Printing error — visual |
| CS-2090 | CS-Missing Print | Printing error — visual |
| CS-2110 | CS-Ink Bleed | Printing error — visual |
| CS-2120 | CS-Alignment Error | Printing error — visual |
| CS-2150 | CS-Partial Print | Printing error — visual |
| CS-2160 | CS-Butterfly Note | Visual ink fold |
| CS-2170 | CS-Gutter Fold | Printing error — visual |
| CS-2180 | CS-Obstruction Error | Printing error — visual |
| CS-2220 | CS-Signature Note | Celebrity/political signature — visual |
| CS-2230 | CS-Stamp Note | Postal/political stamp — visual |
| CS-2250 | CS-Graffiti Note | Markings — visual |
| CS-2260 | CS-Political Stamp | Political overprint — visual |
| CS-2320 | CS-Web Press Error | Production error — visual |
| CS-2330 | CS-Sheet Cutting Error | Production error — visual |
| CS-2340 | CS-Radical Shift | Major misalignment — visual |
| CS-2350 | CS-Experimental Note | Special production notes — visual |

---

## Notes

- CS-227: Appears in book but classification unclear — may be image-only
- CS-60AK (cs_60ak.lua) and CS-Super Radar (cs_super_radar.lua) have no confirmed CS# in the book
- CS-710 (double_year.lua): Verify this is CS-710 vs CS-720 — book may number differently
- CS-810 (cs_count_ones.lua): Verify CS-810 exists in book — agent note from prior batch
- CS-220 and CS-210: Both implement "scattered 40AK" logic; CS-210 (cs_40ak.lua) may actually be CS-190 per book
- Mini Radars (CS-1380–CS-1410) are implemented as palindrome substrings of length 4–7 respectively
- CS-1340 = CS-Mini 3 Radar = CS-Mini 3 Repeater (same pattern, one file)
- CS-1600 (Double Skip Note) = CS-Super Repeater (already implemented as CS-1530)
- CS-1610/1620 are Skip Count notes, NOT ladder variants (TRACKING.md was wrong before batch 3)
- CS-1660 is Stand Alone Pair, NOT Quad Repeater (corrected in batch 3)
- CS-1670/CS-1680 (Stand Alone Triple/Quad) have no confirmed CS# — approximate numbers shown
- Batch 3 added Stand Alone patterns (CS-1650–CS-1740 range)
- Batch 4 added nested/combined group patterns (CS-120, CS-250–CS-470 range) and CS-1880, CS-1990
- CS-250 to CS-470 batch 4 patterns follow a "X inside/within Y" structure with binary or near-binary digit distributions
- Todo descriptions for CS-250–CS-470 were speculative placeholders; actual implementations are named by their true structure
- Batch 5 added bookend variants (CS-980/1000/1010), binary flipper (CS-1050), stand-alone triple/quad/mini-3-radar (CS-1670/1680/1730), centered zeros (CS-1950), quad pairs (CS-60 approx), random double 40AK (CS-240 approx), random quad in triple (CS-270)
- CS-160 (cs_random_double_triples.lua): the un-tagged "CS-Double Triples" (grouped consecutive) examples (#1–#6) appear in the book just before CS-160 with no CS tag number; implemented as cs_double_triples.lua under (no CS#)
- CS-40 (cs_grouped_tri_pairs.lua) was created but removed — book's @CS~40 is all three-pairs-anywhere (superset of CS-50), not the AABBCC-only block I implemented
- CS-60 (cs_grouped_quad_pairs.lua): no @CS~60 tag found; CS number inferred from position between @CS~50 and @CS-70
- CS-240 (cs_double_40ak.lua): no @CS~240 tag found; "CS-Random Double 40AK" described in book without a tag
- CS-1050 = CS-Binary Flipper (2 distinct flip-valid digits); Trinary/Quad Flippers appear later in the book with no @CS tags — implemented in batch 6
- CS-1670/CS-1680 (Stand Alone Triple/Quad) and CS-1730 (Stand Alone Mini 3 Radar) have no @CS tags; numbers are approximate from index listing
- Batch 6 added 10 patterns: CS-Trinary Flipper, CS-Quad Flipper, CS-True Double Quad Binary (CS-920), CS-Random Double Quad Binary (CS-930), CS-Scattered Ladder (CS-1210), CS-Ascending Broken Ladder (CS-1230), CS-Descending Broken Ladder (CS-1240), CS-Pinpoint Radar (CS-1320), CS-Pairs in Quad (CS-330), CS-Stand Alone Year (CS-1810)
- CS-1110/CS-1130/CS-1140 are Rotator subtypes (CS-True Binary Rotator / CS-Trinary Rotator / CS-Quad Rotator); CS-1110 may overlap CS-1040 (True Binary Flipper) since both use flip_string(d)==d — deferred pending book clarification
- CS-1220 (CS-Broken Ladder) is a superset of CS-1230 (Ascending) and CS-1240 (Descending); deferred until subtypes are stable
- CS-1250 (CS-Buildable Ladder) is complex and requires more book research
- CS-1830 (CS-Stand Alone U.S. Date Year) requires calendar logic; moved to Deferred
- CS-1890 (formerly "CS-Near Staircase") was removed from Todo — book research shows this is CS-Stand Alone Mini Up Ladder 5, already covered by CS-1880 (cs_stand_alone_ladder.lua matches ladders of 4-7 digits, including length 5)
