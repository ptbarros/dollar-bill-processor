--[[
Pattern: CS_CENTERED_ZEROS
DisplayName: CS-Centered Zeros
Description: At least one zero at position 4 or 5 (1-indexed, center of the serial), expanding outward from there. Even a single center zero qualifies. e.g., M xxx0xxxx M or M xxxx0xxx M.
BookRef: CS-1950
Tier: 7
Examples: ["12305678", "12300456", "12004567"]
Odds: 1 in 10
Price: $0-$5
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- At least one zero at center position 4 or 5 (1-indexed)
    if d:sub(4, 4) ~= "0" and d:sub(5, 5) ~= "0" then
        return {matched = false}
    end

    -- Find the contiguous zero run that covers the matched center position
    local runs = find_runs(d)
    for _, run in ipairs(runs) do
        if run.digit == "0" then
            local s = run.start + 1          -- 1-indexed start
            local e = run.start + run.length -- 1-indexed end (inclusive)
            if (s <= 4 and e >= 4) or (s <= 5 and e >= 5) then
                local base = run.start
                return {
                    matched = true,
                    group_boxes = {
                        {from = base, to = base + run.length - 1, color = "cyan", thickness = 3}
                    },
                    message = run.length .. " centered zero(s) at position(s) " .. s .. "–" .. e .. " (CS-Centered Zeros)"
                }
            end
        end
    end

    return {matched = false}
end
