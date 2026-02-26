--[[
Pattern: CS_RANDOM_DAY_MONTH
DisplayName: CS-Random Day Month Note
Description: A valid mmdd or ddmm block at any position, with the remaining 4 digits random (not all zeros and not 4OAK).
BookRef: CS-800
Tier: 8
Examples: ["12251357", "45122513", "13122545"]
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    for start = 1, 5 do
        local block = d:sub(start, start + 3)
        local a = tonumber(block:sub(1, 2))
        local b = tonumber(block:sub(3, 4))

        local rest = d:sub(1, start - 1) .. d:sub(start + 4)
        if #rest == 4 then
            -- Exclude True Day (all zeros) and Numbered Day (4OAK non-zero)
            local all_zero = (rest == "0000")
            local first = rest:sub(1, 1)
            local all_same = (first ~= "0" and rest == string.rep(first, 4))
            if not all_zero and not all_same then
                local s0 = start - 1
                local other_pos = {}
                for i = 0, 7 do
                    if i < s0 or i > s0 + 3 then
                        table.insert(other_pos, i)
                    end
                end

                -- Check as mmdd (US/INTL)
                if is_valid_mmdd(a, b) then
                    return {
                        matched = true,
                        group_boxes = {
                            {from = s0, to = s0 + 1, color = "orange", thickness = 2},
                            {from = s0 + 2, to = s0 + 3, color = "coral", thickness = 2}
                        },
                        highlights = {{positions = other_pos, color = "lime"}},
                        message = string.format("Random Day Month: %02d/%02d at position %d", a, b, s0)
                    }
                end

                -- Check as ddmm (EU)
                if is_valid_mmdd(b, a) then
                    return {
                        matched = true,
                        group_boxes = {
                            {from = s0, to = s0 + 1, color = "coral", thickness = 2},
                            {from = s0 + 2, to = s0 + 3, color = "orange", thickness = 2}
                        },
                        highlights = {{positions = other_pos, color = "lime"}},
                        message = string.format("Random Day Month: %02d/%02d (EU) at position %d", a, b, s0)
                    }
                end
            end
        end
    end

    return {matched = false}
end
