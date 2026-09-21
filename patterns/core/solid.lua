--[[
Pattern: SOLID
Description: All 8 digits are identical (e.g., 88888888)
Tier: 1
Examples: ["88888888", "11111111", "00000000"]
Odds: 1 in 12,000,000 (8 per 96M)
Price: $1,000-$10,000+
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check if all digits are the same
    local first = digits:sub(1, 1)
    for i = 2, 8 do
        if digits:sub(i, i) ~= first then
            return {matched = false}
        end
    end

    return {
        matched = true,
        -- Single box around the whole serial (Ed review), no per-digit boxes.
        highlights = {},
        group_boxes = {
            {from = 0, to = 7, color = "orange", thickness = 3}
        },
        connectors = {},
        message = "Perfect solid - all " .. first .. "s"
    }
end
