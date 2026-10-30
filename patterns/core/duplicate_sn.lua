--[[
Pattern: DUPLICATE_SN
DisplayName: Duplicate S/N
Catalog: C-137
Description: A famous U.S. printing error. Serial numbers are supposed to be unique, but the Bureau of Engraving and Printing accidentally printed two separate batches of Series 2013 $1 New York (district "B") STAR notes using the SAME serial numbers -- so two different real notes share one serial. This flags such a note: a Series 2013 bill whose serial starts with B and ends in a star, with a number in the ranges known to have been duplicated. Collectors prize finding both notes of a matching pair.
Tier: 3
Odds: Varies — flags Series 2013 B* star notes in the known duplicate-serial ranges
Examples: ["B00000001*", "B00100000*", "B03200001*", "B05000000*"]
--]]

-- The Series 2013 $1 star-note duplicate: the BEP printed 2013 $1 star notes at
-- two facilities (DC and Fort Worth) and their serial runs OVERLAP, so the same
-- New York (B district) star serial exists on two different notes. It is specific
-- to Series 2013, the B district, and the overlapping ranges below.
function match(ctx)
    local full = ctx.full_serial
    local s = ctx.digits

    -- Must be B prefix (New York district) and * suffix (star note).
    if full:sub(1, 1) ~= "B" or full:sub(-1) ~= "*" then
        return {matched = false}
    end

    -- Must fall in one of the two overlapping (duplicated) serial ranges.
    local num = tonumber(s)
    if not num then
        return {matched = false}
    end
    if not ((num >= 1 and num <= 250000) or (num >= 3200001 and num <= 9600000)) then
        return {matched = false}
    end

    -- Series 2013 requirement (soft gate): the series is only known when plate/
    -- series extraction is on and the OCR succeeded -- it's blank for a typed
    -- serial in Serial Lookup or when extraction is off. So: if we KNOW the series
    -- and it isn't 2013, reject; if we DON'T know it, still flag the note but tell
    -- the user to confirm Series 2013. series_year is a string like "2013"/"2013A".
    local raw = (ctx.metadata and ctx.metadata.series_year) or ""
    local year4 = raw ~= "" and raw:match("%d%d%d%d") or nil
    if year4 and year4 ~= "2013" then
        return {matched = false}
    end
    local confirmed = (year4 == "2013")

    local message
    if confirmed then
        message = "Duplicate serial — Series 2013 New York (B) star note in a duplicated range"
    else
        message = "Potential duplicate serial (New York B* note in a duplicated range) — confirm it's Series 2013"
    end

    return {
        matched = true,
        message = message,
        highlights = {{positions = {0,1,2,3,4,5,6,7}, color = "red"}}
    }
end
