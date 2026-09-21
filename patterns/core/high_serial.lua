--[[
Pattern: HIGH_SERIAL
DisplayName: High Serial
Description: Serial number above 96,000,000 — the circulating maximum for $1 ($20 and below) since the change made partway through Series 1988. Above 99,200,000 is rarer still: that was the ceiling from the early 1980s until then, so those come from a 1970s-or-older series or an uncut collector sheet. Either way, worth checking the note's series.
Tier: 3
Examples: ["96000001", "98765432", "99999999"]
Odds: High serial — an earlier series or an uncut-sheet over-run
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
            message = "High serial " .. digits .. " (over 96,000,000 — an earlier series or a sheet over-run)"
        }
    end

    return {matched = false}
end
