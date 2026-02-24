# Green Guide Pattern Audit

**Rewrite criteria:** Only rewrite if description copies sentence structure/examples verbatim
from the book, or if the Lua logic does not match the book definition.

**Sub-pattern methodology:** When a pattern's description references another pattern by name
(e.g. "contains two CS-Pairs and a CS-3OAK"), look that sub-pattern up in the ODS spreadsheet,
confirm its exact definition, and verify the Lua code enforces that definition for the
sub-component — not just the count. Examples of what to check:
- CS-Pair = two identical digits **adjacent** (pos[2] - pos[1] == 1)
- CS-2OAK = two identical digits **separated** (pos[2] - pos[1] > 1)
- CS-3OAK = three identical digits with max consecutive run < 3
- CS-Triple = three identical digits all consecutive (run length >= 3)
This was discovered in batch 1: CS-120 was missing the pair adjacency check, and CS-160 was
missing the scatter check — both caught and fixed by cross-referencing sub-pattern ODS definitions.

**Resume:** Read this file, find the first row where Status = `pending`, start from that batch.

## Batch History

| Batch | CS# Range | Status |
|-------|-----------|--------|
| 0 | CS-30, CS-110, CS-500, CS-1260, CS-1370 | done — initial review session |
| 1 | CS-50 through CS-190 | done |
| 2 | CS-200 through CS-330 | pending |
| 3 | CS-10/20/40/80/90/140/180/220/300/320/340/350/390/450/970/1030/1080/1500/1510 | pending — batch 7 new patterns |

## Pattern Status

| Status | CS# | Book Name | File | Flags | Notes |
|--------|-----|-----------|------|-------|-------|
| pending | CS-10 | CS-2OAKs | cs_2oaks.lua | clean | |
| pending | CS-20 | CS-Two Pairs | cs_grouped_two_pairs.lua | clean | |
| done | CS-30 | CS-Random Two Pairs | cs_two_pairs.lua | clean | Logic fix: adjacency check added; desc+examples rewritten |
| pending | CS-40 | CS-Tri Pairs | cs_tri_pairs_grouped.lua | clean | |
| done | CS-50 | CS-Random Tri Pairs | cs_tri_pairs.lua | clean | Desc: removed verbatim examples; logic: adjacency check added; example replaced |
| pending | CS-80 | CS-Pairs in Pairs | cs_pairs_in_pairs.lua | clean | |
| pending | CS-90 | CS-Random Pairs in Pairs | cs_random_pairs_in_pairs.lua | clean | |
| done | CS-60 | CS-Quad Pairs | cs_grouped_quad_pairs.lua | clean | Desc+examples: replaced verbatim 11223344 |
| done | CS-70 | CS-Random Quad Pairs | cs_quad_pairs.lua | clean | Desc+examples: replaced both verbatim; logic: adjacency check added |
| done | CS-100 | CS-Triples | cs_triple.lua | clean | DisplayName fixed (Triple→Triples); desc OAK typo fixed |
| done | CS-110 | CS-3OAK | cs_30ak.lua | clean | OAK typo in message fixed |
| done | CS-120 | CS-Paired 30AK | cs_paired_30ak.lua | clean | DisplayName correct (3OAK); desc: OAK typos+verbatim example fixed; logic: scatter check + pair adjacency check added |
| done | CS-130 | CS-Triple Triple Pair | cs_triple_triple_pair.lua | clean | No changes needed |
| pending | CS-140 | CS-Random Triple Triple Pair | cs_random_triple_triple_pair.lua | clean | Logic fix applied: all-consecutive exclusion added |
| done | CS-150 | CS-Doubles Triples | cs_double_triples.lua | clean | DisplayName fixed; desc: removed verbatim example, clarified wording |
| done | CS-160 | CS-Random Double Triples | cs_random_double_triples.lua | clean | Desc: removed verbatim example; logic: scatter check added (each triple must have max run < 3) |
| done | CS-170 | CS-Triple Double Double | cs_triple_double_double.lua | clean | No changes needed |
| pending | CS-180 | CS-Random Triple Double Double | cs_random_triple_double_double.lua | clean | Logic fix applied: all-consecutive exclusion added; examples corrected |
| done | CS-190 | CS-4OAK | cs_40ak.lua | clean | Message OAK typo fixed |
| done | CS-500 | CS-Solids & CS-8OAK | cs_solid.lua | clean | OAK typos fixed (batch 0) |
| done | CS-1260 | CS-Super Radar | cs_super_radar.lua | clean | No changes needed (batch 0) |
| done | CS-1370 | CS-Mini 3 Radar | cs_mini_3_radar.lua | clean | Desc+examples rewritten; message CS# fixed (batch 0) |
| pending | CS-200 | CS-Quad | cs_quad.lua | clean | |
| pending | CS-210 | CS-Random 4OAK | cs_random_40ak.lua | desc-phrases(5), ex-overlap(2) | |
| pending | CS-220 | CS-Leading, Center & Trailing Quads | cs_leading_center_trailing_quads.lua | clean | |
| pending | CS-230 | CS-Double Quad | cs_double_quad.lua | ex-overlap(1) | |
| pending | CS-240 | CS-Random Double 4OAK | cs_double_40ak.lua | clean | |
| pending | CS-250 | CS-Quad in Quad | cs_quad_in_quad.lua | ex-overlap(3) | |
| pending | CS-260 | CS-Quad in Triple | cs_quad_in_triple.lua | clean | |
| pending | CS-270 | CS-Random Quad in Triple | cs_random_quad_in_triple.lua | clean | |
| pending | CS-280 | CS-Double Double | cs_double_double.lua | ex-overlap(1) | |
| pending | CS-290 | CS-Triple in Quad | cs_triple_in_quad.lua | clean | |
| pending | CS-300 | CS-Random Triple in Quad | cs_random_triple_in_quad.lua | clean | Logic fix applied: scattered triple check added; examples corrected |
| pending | CS-310 | CS-Quad and Pairs | cs_quad_and_pairs.lua | ex-overlap(1) | |
| pending | CS-320 | CS-Random Quad and Pairs | cs_random_quad_and_pairs.lua | clean | |
| pending | CS-330 | CS-Pairs in Quad | cs_pairs_in_quad.lua | ex-overlap(1) | |
| pending | CS-340 | CS-Quads and Triples | cs_quads_and_triples.lua | clean | |
| pending | CS-350 | CS-Repeating Doubles | cs_repeating_doubles.lua | clean | |
| pending | CS-360 | CS-5OAK | cs_50ak.lua | clean | |
| pending | CS-370 | CS-Leading, Center, and Trailing Quints | cs_quint.lua | name:"CS-Quint"≠book | |
| pending | CS-380 | CS-Quint in a Pair | cs_quint_in_pair.lua | clean | |
| pending | CS-390 | CS-Pair in a Quint | cs_pair_in_quint.lua | clean | |
| pending | CS-400 | CS-Random Quint and Pair | cs_random_quint_and_pair.lua | clean | |
| pending | CS-410 | CS-Quint in a Triple | cs_quint_in_triple.lua | ex-overlap(2) | |
| pending | CS-420 | CS-Triple in a Quint | cs_triple_in_quint.lua | ex-overlap(3) | |
| pending | CS-430 | CS-6OAK | cs_60ak.lua | clean | |
| pending | CS-440 | CS-Sextup | cs_sextup.lua | clean | |
| pending | CS-450 | CS-Pair and a Sextup | cs_pair_and_sextup.lua | clean | |
| pending | CS-460 | CS-Pair in a Sextup | cs_pair_in_sextup.lua | ex-overlap(3) | |
| pending | CS-470 | CS-Random Pair in a Sextup | cs_random_pair_in_sextup.lua | ex-overlap(1) | |
| pending | CS-480 | CS-Seven | cs_seven.lua | clean | |
| pending | CS-490 | CS-7OAK | cs_70ak.lua | clean | |
| pending | CS-710 | CS-Double Year Note | double_year.lua | clean | |
| pending | CS-810 | CS-Count Ones | cs_count_ones.lua | clean | |
| pending | CS-820 | CS-Count Tens | cs_count_tens.lua | clean | |
| pending | CS-900 | CS-True Binary | cs_true_binary.lua | ex-overlap(1) | |
| pending | CS-910 | CS-Binary | cs_binary.lua | clean | |
| pending | CS-920 | CS-True Double Quad Binary | cs_true_double_quad_binary.lua | ex-overlap(2) | |
| pending | CS-930 | CS-Random Double Quad Binary | cs_random_double_quad_binary.lua | ex-overlap(1) | |
| pending | CS-940 | CS-Trinary | cs_trinary.lua | clean | |
| pending | CS-950 | CS-Single Bookend | cs_single_bookend.lua | desc-phrases(3) | |
| pending | CS-960 | CS-Dual Matched Bookend | cs_dual_bookend.lua | clean | |
| pending | CS-970 | CS-Dual Radar Bookend | cs_dual_radar_bookend.lua | clean | |
| pending | CS-980 | CS-Dual Repeater Bookend | cs_dual_repeater_bookend.lua | clean | |
| pending | CS-990 | CS-Tri Matched Bookend | cs_tri_bookend.lua | clean | |
| pending | CS-1000 | CS-Tri Repeated Bookend | cs_tri_repeated_bookend.lua | clean | |
| pending | CS-1010 | CS-Tri Radar Bookend | cs_tri_radar_bookend.lua | clean | |
| pending | CS-1030 | CS-Unary Flipper | cs_unary_flipper.lua | clean | |
| pending | CS-1040 | CS-True Binary Flipper | cs_true_binary_flipper.lua | clean | |
| pending | CS-1050 | CS-Binary Flipper | cs_binary_flipper.lua | ex-overlap(1) | |
| pending | CS-1060 | CS-Trinary Flipper | cs_trinary_flipper.lua | ex-overlap(1) | |
| pending | CS-1070 | CS-Quad Flipper | cs_quad_flipper.lua | ex-overlap(1) | |
| pending | CS-1080 | CS-Quinary Flipper | cs_quinary_flipper.lua | clean | |
| pending | CS-1090 | CS-Rotators | cs_rotator.lua | name:"CS-Rotator"≠book | |
| pending | CS-1160 | CS-Tetradic | cs_tetradic.lua | ex-overlap(1) | |
| pending | CS-1170 | CS-Ascending Ladder | cs_ascending_ladder.lua | ex-overlap(3) | |
| pending | CS-1180 | CS-Descending Ladder | cs_descending_ladder.lua | ex-overlap(3) | |
| pending | CS-1190 | CS-Ascending Looping Ladder | cs_looping_ladder_asc.lua | clean | |
| pending | CS-1200 | CS-Descending Looping Ladder | cs_looping_ladder_desc.lua | clean | |
| pending | CS-1210 | CS-Scattered Ladder | cs_scattered_ladder.lua | ex-overlap(1) | |
| pending | CS-1230 | CS-Ascending Broken Ladder | cs_ascending_broken_ladder.lua | clean | |
| pending | CS-1240 | CS-Descending Broken Ladder | cs_descending_broken_ladder.lua | clean | |
| pending | CS-1270 | CS-Full Radar | cs_full_radar.lua | ex-overlap(1) | |
| pending | CS-1280 | CS-Bookend Full Radar | cs_bookend_full_radar.lua | ex-overlap(1) | |
| pending | CS-1290 | CS-Wide Radar | cs_wide_radar.lua | ex-overlap(1) | |
| pending | CS-1300 | CS-Split Six Radar | cs_split_six_radar.lua | ex-overlap(1) | |
| pending | CS-1310 | CS-Quad Bookend Radar | cs_quad_bookend_radar.lua | ex-overlap(1) | |
| pending | CS-1320 | CS-Pinpoint Radar | cs_pinpoint_radar.lua | ex-overlap(1) | |
| pending | CS-1330 | CS-Oscillating Radar | cs_oscillating_radar.lua | ex-overlap(1) | |
| pending | CS-1350 | CS-Lucky Seven Radar | cs_lucky_seven_radar.lua | clean | |
| pending | CS-1380 | CS-Mini 4 Radar | cs_mini_4_radar.lua | desc-phrases(2) | |
| pending | CS-1390 | CS-Mini 5 Radar | cs_mini_5_radar.lua | desc-phrases(3) | |
| pending | CS-1400 | CS-Mini 6 Radar | cs_mini_6_radar.lua | desc-phrases(7) | |
| pending | CS-1410 | CS-Mini 7 Radar | cs_mini_7_radar.lua | clean | |
| pending | CS-1480 | CS-Paired Quad Repeater | cs_full_repeater.lua | clean | |
| pending | CS-1500 | CS-Triple Repeater | cs_triple_repeater.lua | clean | |
| pending | CS-1510 | CS-Six Repeater in a Pair | cs_six_repeater_in_pair.lua | clean | |
| pending | CS-1520 | CS-Radar Repeater | cs_radar_repeater.lua | desc-phrases(1), ex-overlap(1) | |
| pending | CS-1530 | CS-Super Repeater | cs_super_repeater.lua | clean | |
| pending | CS-1550 | CS-Mini 4 Repeater | cs_mini_4_repeater.lua | clean | |
| pending | CS-1560 | CS-Mini 5 Repeater | cs_mini_5_repeater.lua | clean | |
| pending | CS-1570 | CS-Mini 6 Repeater | cs_mini_6_repeater.lua | clean | |
| pending | CS-1580 | CS-Mini 7 Repeater | cs_mini_7_repeater.lua | clean | |
| pending | CS-1590 | CS-Single Skip Notes | cs_single_skip_note.lua | desc-phrases(2) | |
| pending | CS-1610 | CS-Skip Count Up Note | cs_skip_count_up.lua | desc-phrases(1) | |
| pending | CS-1620 | CS-Skip Count Down Note | cs_skip_count_down.lua | desc-phrases(1) | |
| pending | CS-1650 | CS-Stand Alone Singles | cs_stand_alone_single.lua | name:"CS-Stand Alone Single"≠book, ex-overlap(3) | |
| pending | CS-1660 | CS-Stand Alone Pair | cs_stand_alone_pair.lua | ex-overlap(3) | |
| pending | CS-1670 | CS-Stand Alone Triple | cs_stand_alone_triple.lua | ex-overlap(1) | |
| pending | CS-1680 | CS-Stand Alone Quad | cs_stand_alone_quad.lua | clean | |
| pending | CS-1690 | CS-Stand Alone Quint | cs_stand_alone_quint.lua | ex-overlap(2) | |
| pending | CS-1710 | CS-Stand Alone Double Repeater | cs_stand_alone_double_repeater.lua | ex-overlap(3) | |
| pending | CS-1720 | CS-Stand Alone Tri Repeater | cs_stand_alone_tri_repeater.lua | clean | |
| pending | CS-1730 | CS-Stand Alone Mini 3 Radar | cs_stand_alone_mini_3_radar.lua | clean | |
| pending | CS-1740 | CS-Stand Alone Mini 4 Radar | cs_stand_alone_mini_4_radar.lua | ex-overlap(3) | |
| pending | CS-1810 | CS-Stand Alone Year | cs_stand_alone_year.lua | ex-overlap(2) | |
| pending | CS-1860 | CS-Stand Alone Mini ladder | cs_stand_alone_ladder.lua | name:"CS-Stand Alone Mini Ladder"≠book | |
| pending | CS-1940 | CS-Leading Zeros | cs_leading_zeros.lua | clean | |
| pending | CS-1950 | CS-Centered Zeros | cs_centered_zeros.lua | clean | |
| pending | CS-1960 | CS-Trailing Zeros | cs_trailing_zeros.lua | clean | |
| pending | CS-1970 | CS-High Notes | cs_high_notes.lua | ex-overlap(1) | |
| pending | CS-1980 | CS-Low Notes | cs_low_notes.lua | ex-overlap(2) | |
| pending | CS-1990 | CS-Million Notes | cs_million_note.lua | ex-overlap(1) | |
