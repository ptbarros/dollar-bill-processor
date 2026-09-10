--[[
Pattern: ESS_KNOWN_SERIALS
DisplayName: KNOWN_SERIALS
Description: Match against known collectible serial numbers
Tier: 1
DataFile: known_serials.csv
Examples: ["12345678", "88888888"]
--]]

function match(ctx)
    -- Check if data was loaded
    if not ctx.data_by_key then
        return {matched = false}
    end

    -- Look up the serial in our known serials database
    local entry = ctx.data_by_key[ctx.digits]
    if entry then
        -- Found a match!
        return {
            matched = true,
            highlights = {highlight_range(0, 7, "gold", "Known serial")},
            message = entry.description .. " - " .. entry.value
        }
    end

    return {matched = false}
end
