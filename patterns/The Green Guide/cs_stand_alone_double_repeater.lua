--[[
Pattern: CS_STAND_ALONE_DOUBLE_REPEATER
DisplayName: CS-Stand Alone Double Repeater
Description: Two CS-2OAKs that alternate (ABAB or ABABAB) within zeros. e.g., M 01212000 M (2 repeats) or M 01212120 M (3 repeats).
BookRef: CS-1710
Tier: 4
Examples: ["01212000", "00121200", "01212120"]
Odds: 1 in 396
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    if d:sub(1, 1) ~= "0" or d:sub(8, 8) ~= "0" then
        return {matched = false}
    end

    -- Try ABAB (4-char block, 2 repeats) and ABABAB (6-char block, 3 repeats)
    for block_len = 4, 6, 2 do
        local max_start = 8 - block_len  -- last valid 1-indexed start (block must end at pos ≤ 7)
        for i = 2, max_start do
            local a = d:sub(i, i)
            local b = d:sub(i + 1, i + 1)

            if a ~= "0" and b ~= "0" and a ~= b then
                -- Check ABAB... pattern for the full block
                local is_repeat = true
                for j = 0, block_len - 1 do
                    local expected = (j % 2 == 0) and a or b
                    if d:sub(i + j, i + j) ~= expected then
                        is_repeat = false
                        break
                    end
                end

                if is_repeat then
                    -- All positions outside the block must be zero
                    local all_zero = true
                    for j = 1, 8 do
                        if j < i or j > i + block_len - 1 then
                            if d:sub(j, j) ~= "0" then
                                all_zero = false
                                break
                            end
                        end
                    end

                    if all_zero then
                        local base = i - 1  -- 0-indexed
                        local a_pos, b_pos = {}, {}
                        for j = 0, block_len - 1 do
                            if j % 2 == 0 then
                                table.insert(a_pos, base + j)
                            else
                                table.insert(b_pos, base + j)
                            end
                        end
                        local repeats = block_len / 2
                        return {
                            matched = true,
                            highlights = {
                                {positions = a_pos, color = "orange"},
                                {positions = b_pos, color = "coral"}
                            },
                            message = string.rep(a .. b, repeats) .. " stand-alone double repeater (" .. repeats .. "× at pos " .. i .. ") (CS-Stand Alone Double Repeater)"
                        }
                    end
                end
            end
        end
    end

    return {matched = false}
end
