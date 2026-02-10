--[[
Pattern: SEAL_SHIFT
DisplayName: Seal Shift
Description: Overprint elements shifted from normal position (any direction)
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
    -- Compares overprint (seal, serials) to intaglio (plate, series year)
    local seal_y = metadata.seal_y
    if not seal_y then
        return {matched = false}
    end

    -- Threshold: |deviation| > 1.5% (pairwise median method)
    -- Based on std dev ~0.81%, this catches shifts ~2 std devs from normal
    local abs_deviation = math.abs(seal_y)
    if abs_deviation <= 1.5 then
        return {matched = false}
    end

    -- Determine direction
    local direction
    if seal_y < 0 then
        direction = string.format("%.1f%% up", -seal_y)
    else
        direction = string.format("%.1f%% down", seal_y)
    end

    return {
        matched = true,
        message = string.format("Overprint shift (%s)", direction)
    }
end
