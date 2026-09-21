--[[
Pattern: HIGH_SERIAL
DisplayName: High Serial
Description: Serial number above 96,000,000, the modern circulating maximum for $1 ($20 and below, Series 1988 onward). A serial this high is either a pre-1988 note (99,999,999 was standard production until the 1970s) or an uncut collector-sheet over-run — either way, worth checking the note's series.
Tier: 3
Examples: ["96000001", "98765432", "99999999"]
Odds: High serial — a pre-1988 note or an uncut-sheet over-run
Price:
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- === Editable threshold ===
    -- 96,000,000 is the circulating maximum for $1 (and most notes); serials above
    -- it are used on uncut collector sheets. Raise this for cases where circulation
    -- went higher: $50/$100 run to 99,200,000, and some early series print sheets
    -- only above ~99.2M-99.84M (1988 / 1985 / 1981). Source: uspapermoney.info.
    local THRESHOLD = 96000000
    -- ==========================

    local n = tonumber(digits)
    if n and n > THRESHOLD then
        return {
            matched = true,
            highlights = {
                highlight({0, 1, 2, 3, 4, 5, 6, 7}, "blue", "serial above 96,000,000")
            },
            message = "High serial " .. digits .. " (over 96,000,000 — pre-1988 note or sheet over-run)"
        }
    end

    return {matched = false}
end
