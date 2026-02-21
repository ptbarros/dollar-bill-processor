--[[
Pattern: CS_SEXTUP
DisplayName: CS-Sextup
Description: Six or more of the same digit grouped consecutively. Scattered version is CS-60AK (CS-410).
BookRef: CS-440
Tier: 2
Examples: ["11111122", "66666600", "00000099"]
Odds: 1 in 1,111,200
Price: $75-$300
--]]

function match(ctx)
    local runs = find_runs(ctx.digits)
    for _, run in ipairs(runs) do
        if run.length >= 6 then
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
