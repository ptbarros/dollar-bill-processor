--[[
Pattern: CS_STAND_ALONE_TRIPLE
DisplayName: CS-Stand Alone Triple
Description: Exactly one run of three consecutive identical non-zero digits, with all other positions being zero, and zeros present on both sides of the triple. e.g., M 00333000 M or M 02220000 M.
BookRef: CS-1670
Tier: 4
Examples: ["00033300", "02220000", "00022200"]
Odds: 1 in 550
Price: $15-$100
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Must start and end with 0 (zeros on both sides)
    if d:sub(1, 1) ~= "0" or d:sub(8, 8) ~= "0" then
        return {matched = false}
    end

    -- Scan for an isolated 3-run of a non-zero digit
    -- Start positions 2–5 (1-indexed) ensure zeros exist on both sides
    for start = 2, 5 do
        local ch = d:sub(start, start)
        if ch ~= "0" and
           d:sub(start + 1, start + 1) == ch and
           d:sub(start + 2, start + 2) == ch then

            -- Verify run length is exactly 3 (not 4+)
            local before = d:sub(start - 1, start - 1)
            local after_pos = start + 3
            local after = after_pos <= 8 and d:sub(after_pos, after_pos) or "X"

            if before ~= ch and after ~= ch then
                -- All other positions must be 0
                local all_others_zero = true
                for j = 1, 8 do
                    if j < start or j >= start + 3 then
                        if d:sub(j, j) ~= "0" then
                            all_others_zero = false
                            break
                        end
                    end
                end
                if all_others_zero then
                    local base = start - 1  -- 0-indexed
                    return {
                        matched = true,
                        group_boxes = {
                            {from = base, to = base + 2, color = "gold", thickness = 3}
                        },
                        message = ch..ch..ch .. " stand-alone triple at position " .. start .. " (CS-Stand Alone Triple)"
                    }
                end
            end
        end
    end

    return {matched = false}
end
