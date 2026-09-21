--[[
Pattern: TRIPLE_DOUBLE
DisplayName: Triple Double Double
Description: A triple and two pairs back to back, in any order, starting at the first or second digit, with one odd digit at the front or the end (e.g. 1·333·44·55).
Tier: 6
Examples: ["13334455", "33344551", "44433551", "55443331"]
Odds: 1 in 2,352 (40,824 per 96M)
Price: $25-$75
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Consecutive runs across all 8 digits.
    local runs = {}
    local i = 1
    while i <= 8 do
        local dch = digits:sub(i, i)
        local len = 1
        while i + len <= 8 and digits:sub(i + len, i + len) == dch do len = len + 1 end
        table.insert(runs, {digit = dch, start = i - 1, length = len})
        i = i + len
    end

    -- Exactly one triple, two pairs and one single (run lengths 3, 2, 2, 1).
    if #runs ~= 4 then return {matched = false} end
    local lengths = {}
    for _, r in ipairs(runs) do table.insert(lengths, r.length) end
    table.sort(lengths, function(a, b) return a > b end)
    if not (lengths[1] == 3 and lengths[2] == 2 and lengths[3] == 2 and lengths[4] == 1) then
        return {matched = false}
    end

    -- The triple and two pairs must sit back to back (7 digits in a row), so the lone
    -- single must be the FIRST or LAST run -- never in the middle (Ed review).
    if runs[1].length ~= 1 and runs[#runs].length ~= 1 then
        return {matched = false}
    end

    -- One colored box per run, but NOT around the lone stray digit (Ed review).
    local colors = {"blue", "orange", "magenta", "red"}
    local group_boxes = {}
    local ci = 1
    for _, r in ipairs(runs) do
        if r.length > 1 then
            table.insert(group_boxes, {from = r.start, to = r.start + r.length - 1,
                color = colors[ci], thickness = 3})
            ci = ci + 1
        end
    end

    return {
        matched = true,
        highlights = {},
        group_boxes = group_boxes,
        connectors = {},
        message = "Triple Double Double"
    }
end
