--[[
Pattern: THREE_IN_A_ROW
DisplayName: 3 in a Row
Catalog: C-134
Description: 3 consecutive identical digits (triple)
Tier: 9
Odds: 1 in 20 (4,746,789 per 96M)
Examples: ["00012345", "12333456", "11123456"]
--]]

function match(ctx)
    local s = ctx.digits

    for start = 1, 6 do
        local digit = s:sub(start, start)
        local all_same = true
        for i = start, start + 2 do
            if s:sub(i, i) ~= digit then
                all_same = false
                break
            end
        end

        if all_same then
            -- Maximal run of exactly 3: a longer run belongs to 4/5/6/7-in-a-row,
            -- so stand aside if the triple extends either side (mirrors
            -- four_in_a_row).
            local extends_before = (start > 1 and s:sub(start - 1, start - 1) == digit)
            local extends_after = (start + 3 <= 8 and s:sub(start + 3, start + 3) == digit)

            if not extends_before and not extends_after then
                local positions = {}
                for i = start - 1, start + 1 do
                    table.insert(positions, i)
                end
                return {
                    matched = true,
                    message = "Triple " .. digit .. "s",
                    highlights = {},
                    group_boxes = {{from = positions[1], to = positions[#positions], color = "orange", thickness = 3}}
                }
            end
        end
    end

    return {matched = false}
end
