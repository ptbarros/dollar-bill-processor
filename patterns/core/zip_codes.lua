--[[
Pattern: ZIP_CODE
DisplayName: Zip Code (Zero-Bordered)
Description: A valid US ZIP code as 5 consecutive digits with a 0 immediately before and after it (0 ZZZZZ 0)
Tier: 5
Examples: ["09021005", "50100010", "06060105"]
Odds: ~1 in 150
Price: $15-$40
DataFile: zip_codes.csv
--]]

-- The serial's 8 digits contain a valid 5-digit US ZIP code that is bordered by
-- a 0 on each side. A 5-digit run flanked by a 0 before and after only fits at
-- two places in 8 digits: the ZIP starting at position 2 (0 ZZZZZ 0 _) or
-- position 3 (_ 0 ZZZZZ 0). ZIP validity is a lookup into zip_codes.csv
-- (~31,900 real US ZIPs) via ctx.data_by_key, keyed by the 5-digit code.

function match(ctx)
    -- data_by_key is the ZIP set keyed by code; absent if the CSV failed to load.
    if not ctx.data_by_key then
        return {matched = false}
    end

    local d = ctx.digits
    if #d ~= 8 then
        return {matched = false}
    end

    -- z0 = 0-indexed start of the 5-digit ZIP run (1 or 2 leave room for a
    -- bordering 0 on both sides within the 8 digits).
    for _, z0 in ipairs({1, 2}) do
        local zip    = d:sub(z0 + 1, z0 + 5)   -- ZIP run, 0-indexed [z0, z0+4]
        local before = d:sub(z0,     z0)       -- digit just before it (0-indexed z0-1)
        local after  = d:sub(z0 + 6, z0 + 6)   -- digit just after it  (0-indexed z0+5)

        if before == "0" and after == "0" and ctx.data_by_key[zip] then
            return {
                matched = true,
                -- Box the ZIP; mute the two bordering zeros.
                group_boxes = {{from = z0, to = z0 + 4, color = "gold", thickness = 3}},
                highlights = {
                    highlight({z0 - 1, z0 + 5}, "gray", "border 0")
                },
                message = "ZIP code " .. zip
            }
        end
    end

    return {matched = false}
end
