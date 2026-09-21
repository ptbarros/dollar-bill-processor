--[[
Pattern: HIGH_SERIAL
DisplayName: High Serial
Description: Serial number above 96,000,000. Circulating $1 serials stop at 96,000,000 (a block is 15 press runs of 6,400,000). Serials above that are printed for uncut collector sheets, which are sold above face value and rarely cut apart — so a circulated note this high came from a separated sheet, making it an uncommon find.
Tier: 3
Examples: ["96000001", "98765432", "99999999"]
Odds: Over-run serial — printed for uncut sheets, rarely circulated
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
            message = "High serial " .. digits .. " (over 96,000,000 — sheet over-run)"
        }
    end

    return {matched = false}
end
