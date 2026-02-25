--[[
Pattern: CS_STAND_ALONE_DOWN_LADDER_6
DisplayName: CS-Stand Alone Mini Down Ladder 6
Description: Six digits grouped together that count down (each = prev-1, mod-10 wrap), with zeros in positions 1 and 8. e.g., M 06543210 M.
BookRef: CS-1920
Tier: 4
Examples: ["06543210"]
Odds: 1 in 21
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Positions 0 and 7 must be zero
    if d:sub(1, 1) ~= "0" or d:sub(8, 8) ~= "0" then
        return {matched = false}
    end

    -- 6 digits at positions 1-6 (1-indexed: 2-7), all must be non-zero
    for i = 2, 7 do
        if d:sub(i, i) == "0" then
            return {matched = false}
        end
    end

    -- Check descending: each digit = (prev - 1) % 10
    for i = 3, 7 do
        local prev = tonumber(d:sub(i - 1, i - 1))
        local curr = tonumber(d:sub(i, i))
        if curr ~= (prev - 1 + 10) % 10 then
            return {matched = false}
        end
    end

    return {
        matched = true,
        group_boxes = {
            {from = 1, to = 6, color = "lime", thickness = 3}
        },
        message = "6-digit descending ladder stand-alone (CS-Stand Alone Mini Down Ladder 6)"
    }
end
