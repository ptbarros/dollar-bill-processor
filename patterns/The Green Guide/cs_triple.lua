--[[
Pattern: CS_TRIPLE
DisplayName: CS-Triple
Description: Three or more of the same digit grouped consecutively. Scattered version is CS-30AK (CS-110).
BookRef: CS-100
Tier: 5
Examples: ["00033300", "11100011", "44478900"]
Odds: 1 in 1,143
Price: $5-$20
--]]

function match(ctx)
    local runs = find_runs(ctx.digits)
    -- Find the first run of length >= 3
    for _, run in ipairs(runs) do
        if run.length >= 3 then
            return {
                matched = true,
                group_boxes = {
                    {from = run.start, to = run.start + run.length - 1, color = "gold", thickness = 2}
                },
                message = run.length .. " consecutive " .. run.digit .. "s"
            }
        end
    end
    return {matched = false}
end
