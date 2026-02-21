--[[
Pattern: CS_SOLID
DisplayName: CS-Solid (CS-80AK)
Description: All 8 digits identical (CS-80AK in book nomenclature). The rarest possible serial.
BookRef: CS-500
Tier: 1
Examples: ["00000000", "11111111", "88888888"]
Odds: 1 in 11,111,111
Price: $2,000-$10,000+
--]]

function match(ctx)
    local d = ctx.digits
    local first = d:sub(1, 1)
    for i = 2, 8 do
        if d:sub(i, i) ~= first then
            return {matched = false}
        end
    end

    return {
        matched = true,
        group_boxes = {
            {from = 0, to = 7, color = "gold", thickness = 3}
        },
        message = "Solid " .. first .. "s — CS-Solid (CS-80AK)"
    }
end
