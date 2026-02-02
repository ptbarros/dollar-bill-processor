--[[
Pattern: STEP_LADDER
Description: Steps of 2 (02468xxx or similar)
Tier: 4
Examples: ["02468135", "13579246"]
Odds: 1 in 5,000
Price: $5-$30
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check for step ladder with step of 2 (at least 4 digits)
    local best_length = 0
    local best_start = 0

    for start = 1, 5 do  -- Start positions 0-4 (1-indexed)
        local length = 1
        local d = tonumber(digits:sub(start, start))
        for i = start + 1, 8 do
            local next_d = tonumber(digits:sub(i, i))
            local expected = (d + (i - start) * 2) % 10
            if next_d == expected then
                length = length + 1
            else
                break
            end
        end
        if length > best_length then
            best_length = length
            best_start = start - 1  -- Convert to 0-indexed
        end
    end

    if best_length < 4 then
        return {matched = false}
    end

    -- Highlight the step ladder
    local positions = {}
    for i = 0, best_length - 1 do
        table.insert(positions, best_start + i)
    end

    return {
        matched = true,
        highlights = {
            highlight(positions, "lime", "step ladder")
        },
        connectors = {},
        message = best_length .. "-digit step ladder (step 2)"
    }
end
