--[[
Pattern: CS_SEVEN
DisplayName: CS-Seven
Description: Seven consecutive identical digits. Scattered version is CS-70AK (CS-510).
BookRef: CS-480
Tier: 1
Examples: ["11111112", "33333330", "77777771"]
Odds: 1 in 11,111,111
Price: $500-$2,000+
--]]

function match(ctx)
    local runs = find_runs(ctx.digits)
    for _, run in ipairs(runs) do
        if run.length >= 7 then
            return {
                matched = true,
                group_boxes = {
                    {from = run.start, to = run.start + run.length - 1, color = "gold", thickness = 3}
                },
                message = run.length .. " consecutive " .. run.digit .. "s"
            }
        end
    end
    return {matched = false}
end
