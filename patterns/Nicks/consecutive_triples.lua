--[[
Pattern: NICKS_CONSECUTIVE_TRIPLES
DisplayName: Consecutive Triples
Description: Two consecutive triples (AAABBB) with no digits in between
Tier: 3
Examples: ["11122234", "12333444", "00011134", "99988834"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Look for AAABBB pattern
    for start = 1, 3 do
        local seg = s:sub(start, start + 5)

        -- First triple
        if seg:sub(1, 1) == seg:sub(2, 2) and seg:sub(2, 2) == seg:sub(3, 3) then
            -- Second triple
            if seg:sub(4, 4) == seg:sub(5, 5) and seg:sub(5, 5) == seg:sub(6, 6) then
                -- Must be different digits
                if seg:sub(1, 1) ~= seg:sub(4, 4) then
                    return {
                        matched = true,
                        message = "Consecutive triples: " .. seg:sub(1, 1) .. seg:sub(1, 1) .. seg:sub(1, 1) .. " + " .. seg:sub(4, 4) .. seg:sub(4, 4) .. seg:sub(4, 4),
                        group_boxes = {
                            {from = start - 1, to = start + 1, color = "orange"},
                            {from = start + 2, to = start + 4, color = "cyan"}
                        }
                    }
                end
            end
        end
    end

    return {matched = false}
end
