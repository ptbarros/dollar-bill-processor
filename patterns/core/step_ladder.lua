--[[
Pattern: STEP_LADDER
DisplayName: Step Ladder
Description: The serial splits into two 4-digit halves that are identical except one digit, which steps up or down by 1 (e.g. 5191|5091). "Step Down" when the back half is lower, "Step Up" when higher.
Tier: 4
Examples: ["51915091", "77367636", "44784578", "52114211"]
Odds: 1 in 1,500
Price: $5-$30
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Split into two 4-digit halves and compare digit-by-digit.
    -- Require EXACTLY one position to differ, and by exactly 1 (a single "rung").
    -- Comparing digits (not the numeric difference) correctly rejects borrow
    -- cases like 2000 vs 1999 (numeric diff of 1, but all four digits change).
    local diff_pos = nil
    local diff_count = 0
    local direction = nil
    for j = 1, 4 do
        local a = tonumber(digits:sub(j, j))          -- first-half digit
        local b = tonumber(digits:sub(j + 4, j + 4))  -- second-half digit
        if a ~= b then
            if math.abs(a - b) ~= 1 then
                return {matched = false}  -- changed digit isn't a single step
            end
            diff_count = diff_count + 1
            diff_pos = j
            direction = (b > a) and "Up" or "Down"
        end
    end

    if diff_count ~= 1 then
        return {matched = false}
    end

    local p = diff_pos - 1  -- 0-indexed position of the stepped digit in half 1

    return {
        matched = true,
        highlights = {
            highlight({p}, "orange", "step"),
            highlight({p + 4}, "orange", "step"),
        },
        connectors = {
            {from = p, to = p + 4, color = "orange", style = "arc"}
        },
        group_boxes = {
            {from = 0, to = 3, color = "gold", thickness = 3},
            {from = 4, to = 7, color = "gold", thickness = 3},
        },
        message = "Step Ladder (Step " .. direction .. "): back half steps one rung from the front half"
    }
end
