# Handoff Document for New Chat Session

**Copy this entire document into your new chat with Claude Code**

---

## Project Context

I'm continuing development on a dollar bill serial number processing system. We have a working v1.0 and now need to build Phase 1 of the production pipeline.

**Project Location:**
```
/home/pbarros/Pictures/Canon Test Scan/dollar-bill-processor/
```

---

## What's Already Working (v1.0)

We have a **hybrid YOLOv8 + EasyOCR system** that:
- ✅ Extracts serial numbers from scanned dollar bills
- ✅ 100% detection rate (10/10 test bills)
- ✅ 176 bills/minute processing speed
- ✅ Detects 6 fancy number types (repeater, radar, solid, ladder, low serial, binary)
- ✅ Handles star notes (asterisk serials)

**Key Files:**
- `process_bills_yolo.py` - Main processing script
- `best.pt` - Trained YOLOv8 model
- `reference_bill.jpg` - For template alignment
- `README.md` - Full documentation

**Test it:** `python process_bills_yolo.py best.pt`

---

## What We're Building Now (Phase 1)

**Goal:** Automate my father-in-law's workflow for processing 1,000+ bills for eBay listings.

### Requirements

1. **Scanner-Agnostic Input**
   - Process directory with mixed scans (any naming convention)
   - Auto-detect fronts (have serials) vs backs (no serials)
   - Smart pairing: Assume front immediately followed by its back
   - Handle both naming schemes:
     - Scanner 1: `001.jpg` (front), `002.jpg` (back)
     - Scanner 2: `001.jpg` (front), `001_b.jpg` (back)

2. **Fancy Number Filtering**
   - Process ALL bills for serial extraction
   - Only CREATE CROPS for fancy/matching bills
   - Built-in patterns: repeater, radar, solid, ladder, low serial, binary, star notes
   - (Future: Custom patterns via YAML config)

3. **Automated Cropping** (For Fancy Bills Only)
   - Create 10 crops per bill using percentage-based coordinates
   - Crops needed (based on `crop_bills_v5.bat` @"../10 Scans/crop_bills_v5.bat"):
     - Front: seal, full, left third, center third, right third
     - Back: seal, full, left third, center third, right third
   - File naming: `SERIAL_01.jpg` through `SERIAL_10.jpg`
   - Example: `F16936637I_01.jpg`, `F16936637I_02.jpg`, etc.

4. **Stack Position Tracking**
   - Track physical position of each bill in scanned stack
   - Report: "Fancy bills found at positions: 7, 23, 41"
   - Helps locate bills in physical stack

5. **Output Structure**
   ```
   fancy_bills/
     F16936637I_01.jpg
     F16936637I_02.jpg
     ...
     F16936637I_10.jpg
     B12341234G_01.jpg (repeater)
     ...

   results.csv              # All bills processed
   summary.txt              # Fancy bill positions
   non_fancy_files.txt      # List for optional deletion
   ```

6. **Batch Cleanup Helper**
   - Generate list of non-fancy scan files
   - User can optionally delete to save space
   - (Don't auto-delete, just list them)

### Crop Coordinates Reference

From `crop_bills_v5.bat`:
```
Front Seal:   470x440+1085+150   (WIDTHxHEIGHT+XOFFSET+YOFFSET)
Front Left:   710x820+0+0
Front Center: 595x820+655+0
Front Right:  690x820+1160+0

Back Seal:    482x457+1146+181
Back Left:    710x820+0+0
Back Center:  595x820+655+0
Back Right:   690x820+1160+0
```

**Need to convert to percentages** for scanner independence.

### Current Workflow Limitations

Father-in-law currently:
1. Withdraws 1,000 bills from bank
2. Manually slides each under webcam with Google Vision
3. Sets aside fancy numbers
4. Scans fancy bills (front + back)
5. Manually crops 10 images per bill for eBay
6. Posts to eBay

**We're automating:** Steps 2-5 into batch scanning with automated fancy detection and cropping.

---

## Technical Notes

### Front/Back Detection Strategy
```python
# Fronts: Have 2 serial numbers detected
# Backs: Have 0 serial numbers detected
# Already have YOLO detection working!
```

### Percentage-Based Cropping
```python
# Convert pixel coordinates to percentages
# Example: 470x440+1085+150 on 1850x820 image
# = {x: 0.587, y: 0.183, w: 0.254, h: 0.537}
# This works across any scanner resolution
```

### Star Note Format
- Normal: `F16936637I`
- Star note: `F16936637*` (asterisk replaces final letter)
- Already handled by pattern: `[A-Z]\d{8}[A-Z*]`

---

## What to Build

**Main Script:** Extend `process_bills_yolo.py` or create new `process_production.py`

**Features:**
1. Front/back detection function
2. Pairing logic (sequential)
3. Percentage-based crop function
4. Fancy-only filter
5. Position tracking
6. Output generation

**Command-line interface:**
```bash
python process_production.py /path/to/scans --output fancy_bills/
```

---

## Questions to Resolve

1. Should we extend `process_bills_yolo.py` or create new script?
2. Calculate exact percentage coordinates from pixel values?
3. Test with mixed scanner formats?

---

## Next Steps

1. Review existing code: `process_bills_yolo.py`
2. Implement front/back detection
3. Add percentage-based cropping
4. Test on sample data
5. Iterate based on results

---

**Project is ready for Phase 1 development. All working code and documentation is in place.**
