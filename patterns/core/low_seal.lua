--[[
Pattern: LOW_SEAL
DisplayName: Low Seal
Description: Overprint elements shifted down from normal position
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
    -- Positive = overprint shifted DOWN relative to intaglio
    local seal_y = metadata.seal_y
    if not seal_y then
        return {matched = false}
    end

    -- Low seal threshold: deviation > +1.3% (pairwise median method)
    -- Based on std dev ~0.81%, this catches shifts ~1.6 std devs from normal
    if seal_y <= 1.3 then
        return {matched = false}
    end

    return {
        matched = true,
        message = string.format("Low seal (shifted %.1f%% down)", seal_y)
    }
end
