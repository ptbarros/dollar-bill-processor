--[[
Pattern: CS_STAND_ALONE_QUAD
DisplayName: CS-Stand Alone Quad
Description: Exactly one run of four consecutive identical non-zero digits, with all other positions being zero, and zeros present on both sides of the quad. e.g., M 00444400 M or M 02222000 M.
BookRef: CS-1680
Tier: 3
Examples: ["00222200", "02222000", "00022220"]
Odds: 1 in 2,700
Price: $25-$200
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Must start and end with 0 (zeros on both sides)
    if d:sub(1, 1) ~= "0" or d:sub(8, 8) ~= "0" then
        return {matched = false}
    end

    -- Scan for an isolated 4-run of a non-zero digit
    -- Start positions 2–4 (1-indexed) ensure zeros exist on both sides
    for start = 2, 4 do
        local ch = d:sub(start, start)
        if ch ~= "0" and
           d:sub(start + 1, start + 1) == ch and
           d:sub(start + 2, start + 2) == ch and
           d:sub(start + 3, start + 3) == ch then

            -- Verify run length is exactly 4 (not 5+)
            local before = d:sub(start - 1, start - 1)
            local after_pos = start + 4
            local after = after_pos <= 8 and d:sub(after_pos, after_pos) or "X"

            if before ~= ch and after ~= ch then
                -- All other positions must be 0
                local all_others_zero = true
                for j = 1, 8 do
                    if j < start or j >= start + 4 then
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
                            {from = base, to = base + 3, color = "gold", thickness = 3}
                        },
                        message = ch..ch..ch..ch .. " stand-alone quad at position " .. start .. " (CS-Stand Alone Quad)"
                    }
                end
            end
        end
    end

    return {matched = false}
end
