--[[
Pattern: CS_COUNT_ONES
DisplayName: CS-Count by Ones
Description: Serial counts sequentially by 1 in two-digit pairs: 01020304, 02030405, 12131415, etc. Four consecutive two-digit pairs increasing by 1.
BookRef: CS-810
Tier: 3
Examples: ["01020304", "05060708", "12131415"]
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Check ascending by 1
    local result = is_counting_pairs(d, 1)
    if result and result.matched then
        local start_val = result.start_value
        return {
            matched = true,
            group_boxes = {
                {from = 0, to = 1, color = "cyan", thickness = 2},
                {from = 2, to = 3, color = "cyan", thickness = 2},
                {from = 4, to = 5, color = "cyan", thickness = 2},
                {from = 6, to = 7, color = "cyan", thickness = 2},
            },
            connectors = {
                {from = 1, to = 2, color = "lime", style = "line"},
                {from = 3, to = 4, color = "lime", style = "line"},
                {from = 5, to = 6, color = "lime", style = "line"},
            },
            message = string.format("Counts by 1s: %02d %02d %02d %02d (CS-810)", start_val, start_val+1, start_val+2, start_val+3)
        }
    end

    -- Check descending by 1
    local result_desc = is_counting_pairs(d, -1)
    if result_desc and result_desc.matched then
        local start_val = result_desc.start_value
        return {
            matched = true,
            group_boxes = {
                {from = 0, to = 1, color = "cyan", thickness = 2},
                {from = 2, to = 3, color = "cyan", thickness = 2},
                {from = 4, to = 5, color = "cyan", thickness = 2},
                {from = 6, to = 7, color = "cyan", thickness = 2},
            },
            connectors = {
                {from = 1, to = 2, color = "coral", style = "line"},
                {from = 3, to = 4, color = "coral", style = "line"},
                {from = 5, to = 6, color = "coral", style = "line"},
            },
            message = string.format("Counts down by 1s: %02d %02d %02d %02d (CS-810)", start_val, start_val-1, start_val-2, start_val-3)
        }
    end

    return {matched = false}
end
