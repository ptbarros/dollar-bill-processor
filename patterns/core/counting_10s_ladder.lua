--[[
Pattern: COUNTING_10S_LADDER
DisplayName: Count By 10s Ladder
Description: Reads as four two-digit numbers climbing (or dropping) by ten each time (e.g. 10·20·30·40 or 40·30·20·10).
Tier: 3
Odds: 1 in 705,882 (136 per 96M)
Examples: ["10203040", "40302010", "21314151", "51413121"]
Price: $100-$1,000+
--]]

-- Merged 2026-10-14 (Ed review): this was ascending-only and was a strict subset
-- of "Count by 10" (COUNT_TENS), which did ascending AND descending -- so an
-- ascending step-10 serial showed two near-identical cards. Per Paul's call we keep
-- this name ("Count By 10s Ladder", consistent with the other Count By patterns)
-- but adopt the asc+desc behavior, and COUNT_TENS was removed. Odds recomputed for
-- the merged (asc+desc) match set: 140 serials exist, 136 fall in the [1, 96M]
-- counting range (the 4 descending runs starting 96..99 exceed 96M).
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
            message = string.format("Counts by 10s: %02d %02d %02d %02d", start_val, start_val+10, start_val+20, start_val+30)
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
            message = string.format("Counts down by 10s: %02d %02d %02d %02d", start_val, start_val-10, start_val-20, start_val-30)
        }
    end

    return {matched = false}
end
