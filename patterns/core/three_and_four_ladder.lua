--[[
Pattern: THREE_AND_FOUR_LADDER
Description: A 3-digit ladder and a 4-digit ladder, in either order, sitting in the serial (e.g. 123·8765·0).
Tier: 4
Examples: ["12387650", "32145670", "56781230"]
Odds: 1 in 4,400
Price: $10-$100+
--]]

-- Collect maximal monotonic runs (each step +1 or -1) of length >= min_len,
-- as {start (0-indexed), len}. Ascending and descending runs are both returned.
local function mono_runs(s)
    local res = {}
    for _, step in ipairs({1, -1}) do
        local i = 1
        while i <= 8 do
            local j = i
            -- mod-10 so a ladder wrapping 9<->0 reads as one run, consistent with
            -- Three And Five (Ed review): otherwise a wrapping 5-run looked like a
            -- 4-run + stray and this pattern poached clean 3+5 serials.
            while j < 8 and tonumber(s:sub(j + 1, j + 1)) == (tonumber(s:sub(j, j)) + step) % 10 do
                j = j + 1
            end
            if j > i then
                table.insert(res, {start = i - 1, len = j - i + 1})
                i = j + 1
            else
                i = i + 1
            end
        end
    end
    return res
end

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local runs = mono_runs(digits)

    -- Find a run of exactly 4 and a DISJOINT run of exactly 3, in either order
    -- (Ed review: the order of the 3- and 4-digit ladders is interchangeable).
    -- Exact lengths keep a clean 3+5 split as the dedicated Three And Five Ladder
    -- and keep a single long ladder out.
    local r3, r4
    for _, a in ipairs(runs) do
        if a.len == 4 then
            local a1, a2 = a.start, a.start + a.len - 1
            for _, b in ipairs(runs) do
                if b.len == 3 then
                    local b1, b2 = b.start, b.start + b.len - 1
                    if a2 < b1 or b2 < a1 then  -- disjoint
                        r4, r3 = a, b
                        break
                    end
                end
            end
        end
        if r4 then break end
    end

    if not (r3 and r4) then
        return {matched = false}
    end

    return {
        matched = true,
        -- One box around each ladder, no per-digit boxes (Ed review).
        highlights = {},
        group_boxes = {
            {from = r4.start, to = r4.start + r4.len - 1, color = "blue", thickness = 3},
            {from = r3.start, to = r3.start + r3.len - 1, color = "orange", thickness = 3}
        },
        connectors = {},
        message = "3-ladder + 4-ladder"
    }
end
