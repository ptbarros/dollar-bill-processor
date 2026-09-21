--[[
Pattern: PYRAMID_LADDER
DisplayName: Pyramid Ladder
Description: All 8 digits climb by 1 to a single peak near the middle, then fall by 1 to the end, with the peak on the fourth or fifth digit (e.g. 34565432, 23454321).
Tier: 2
Odds: 1 in 8,000,000 (12 per 96M)
Price: $10-$50
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    local n = {}
    for i = 1, 8 do
        n[i] = tonumber(d:sub(i, i))
        if not n[i] then return {matched = false} end
    end

    -- Walk the 8 digits: strictly +1 up to a single peak, then strictly -1.
    local ascending = true
    local peak = nil  -- 0-indexed position of the peak
    for i = 1, 7 do
        local diff = n[i + 1] - n[i]
        if ascending then
            if diff == 1 then
                -- keep climbing
            elseif diff == -1 then
                ascending = false
                peak = i - 1  -- n[i] (position i-1, 0-indexed) is the peak
            else
                return {matched = false}
            end
        else
            if diff ~= -1 then return {matched = false} end
        end
    end
    -- Need both an ascending and a descending leg (peak strictly inside).
    if ascending or peak == nil or peak == 0 then return {matched = false} end

    -- Peak must sit on the 4th or 5th digit (Ed review): index 3 or 4.
    if peak ~= 3 and peak ~= 4 then return {matched = false} end

    local asc_pos, desc_pos = {}, {}
    for i = 0, peak - 1 do table.insert(asc_pos, i) end
    for i = peak + 1, 7 do table.insert(desc_pos, i) end

    return {
        matched = true,
        message = "Pyramid ladder (peak " .. n[peak + 1] .. ")",
        highlights = {
            {positions = asc_pos, color = "blue"},
            {positions = {peak}, color = "orange"},
            {positions = desc_pos, color = "magenta"}
        },
        connectors = {}
    }
end
