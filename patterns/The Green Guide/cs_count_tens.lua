--[[
Pattern: CS_COUNT_TENS
DisplayName: CS-Count by Tens
Description: Serial counts by 10 in two-digit pairs: 10203040 or 40302010. The four pairs are 10, 20, 30, 40 (or reverse).
BookRef: CS-820
Tier: 3
Examples: ["10203040", "40302010", "20304050"]
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Check ascending by 10
    local result = is_counting_pairs(d, 10)
    if result and result.matched then
        local start_val = result.start_value
        return {
            matched = true,
            group_boxes = {
                {from = 0, to = 1, color = "orange", thickness = 2},
                {from = 2, to = 3, color = "orange", thickness = 2},
                {from = 4, to = 5, color = "orange", thickness = 2},
                {from = 6, to = 7, color = "orange", thickness = 2},
            },
            connectors = {
                {from = 1, to = 2, color = "lime", style = "line"},
                {from = 3, to = 4, color = "lime", style = "line"},
                {from = 5, to = 6, color = "lime", style = "line"},
            },
            message = string.format("Counts by 10s: %02d %02d %02d %02d (CS-820)", start_val, start_val+10, start_val+20, start_val+30)
        }
    end

    -- Check descending by 10
    local result_desc = is_counting_pairs(d, -10)
    if result_desc and result_desc.matched then
        local start_val = result_desc.start_value
        return {
            matched = true,
            group_boxes = {
                {from = 0, to = 1, color = "orange", thickness = 2},
                {from = 2, to = 3, color = "orange", thickness = 2},
                {from = 4, to = 5, color = "orange", thickness = 2},
                {from = 6, to = 7, color = "orange", thickness = 2},
            },
            connectors = {
                {from = 1, to = 2, color = "coral", style = "line"},
                {from = 3, to = 4, color = "coral", style = "line"},
                {from = 5, to = 6, color = "coral", style = "line"},
            },
            message = string.format("Counts down by 10s: %02d %02d %02d %02d (CS-820)", start_val, start_val-10, start_val-20, start_val-30)
        }
    end

    return {matched = false}
end
