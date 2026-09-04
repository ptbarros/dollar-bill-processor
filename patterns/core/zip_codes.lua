--[[
Pattern: ZIP_CODE
DisplayName: Zip Code (Zeros)
Description: A valid US ZIP code as 5 consecutive digits, with every other digit a 0 (e.g. 0 ZZZZZ 00)
Tier: 4
Examples: ["90210000", "09021000", "00902100", "00090210"]
Odds: ~1 in 800
Price: $20-$60
DataFile: zip_codes.csv
--]]

-- The 8-digit serial is a valid 5-digit US ZIP code with ALL THREE of the other
-- digits being 0. The ZIP run can sit at four places (0-indexed start 0..3):
--   ZZZZZ000 / 0ZZZZZ00 / 00ZZZZZ0 / 000ZZZZZ
-- ZIP validity is a lookup into zip_codes.csv (~31,900 real US ZIPs) via
-- ctx.data_by_key, keyed by the 5-digit code.

function match(ctx)
    -- data_by_key is the ZIP set keyed by code; absent if the CSV failed to load.
    if not ctx.data_by_key then
        return {matched = false}
    end

    local d = ctx.digits
    if #d ~= 8 then
        return {matched = false}
    end

    for z0 = 0, 3 do
        local zip = d:sub(z0 + 1, z0 + 5)   -- ZIP run, 0-indexed [z0, z0+4]
        if ctx.data_by_key[zip] then
            -- Every digit NOT in the ZIP run must be 0.
            local all_zero = true
            local zero_positions = {}
            for i = 0, 7 do
                if i < z0 or i > z0 + 4 then
                    if d:sub(i + 1, i + 1) ~= "0" then
                        all_zero = false
                        break
                    end
                    table.insert(zero_positions, i)
                end
            end
            if all_zero then
                return {
                    matched = true,
                    group_boxes = {{from = z0, to = z0 + 4, color = "gold", thickness = 3}},
                    highlights = {highlight(zero_positions, "gray")},
                    message = "ZIP code " .. zip
                }
            end
        end
    end

    return {matched = false}
end
