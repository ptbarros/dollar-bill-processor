--[[
Pattern: HIGH_SEAL
DisplayName: High Seal
Description: Overprint elements shifted up from normal position
Tier: 5
Examples: []
Odds: Rare - printing variance
Price: $5-$50+
--]]

function match(ctx)
    local metadata = ctx.metadata
    if not metadata then
        return {matched = false}
    end

    -- seal_y is deviation percentage from expected overprint vs intaglio shift
    -- Negative = overprint shifted UP relative to intaglio
    local seal_y = metadata.seal_y
    if not seal_y then
        return {matched = false}
    end

    -- High seal threshold: deviation < -1.7% (pairwise median method)
    -- Based on std dev ~0.81%, this catches shifts ~2 std devs from normal
    if seal_y >= -1.7 then
        return {matched = false}
    end

    return {
        matched = true,
        message = string.format("High seal (shifted %.1f%% up)", -seal_y)
    }
end
