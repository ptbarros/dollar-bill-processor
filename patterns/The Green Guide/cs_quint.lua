--[[
Pattern: CS_QUINT
DisplayName: CS-Quint
Description: Five or more of the same digit grouped consecutively. Scattered version is CS-50AK (CS-310).
BookRef: CS-370
Tier: 3
Examples: ["11111234", "55555678", "00000123"]
Odds: 1 in 111,120
Price: $25-$100
--]]

function match(ctx)
    local runs = find_runs(ctx.digits)
    for _, run in ipairs(runs) do
        if run.length >= 5 then
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
