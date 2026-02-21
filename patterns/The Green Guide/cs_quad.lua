--[[
Pattern: CS_QUAD
DisplayName: CS-Quad
Description: Four or more of the same digit grouped consecutively. Scattered version is CS-40AK (CS-210).
BookRef: CS-200
Tier: 4
Examples: ["11112345", "44445678", "00001234"]
Odds: 1 in 11,112
Price: $10-$50
--]]

function match(ctx)
    local runs = find_runs(ctx.digits)
    for _, run in ipairs(runs) do
        if run.length >= 4 then
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
