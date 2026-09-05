--[[
Pattern: SEAL_SHIFT
DisplayName: Seal Shift
Description: Overprint elements shifted from normal position
Overlay: none
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

    -- seal_x/seal_y are center-to-center offset as % of ONE_hashed dimensions
    -- Standard coordinates: +x = right, +y = up
    local seal_x = metadata.seal_x or 0
    local seal_y = metadata.seal_y or 0
    local containment = metadata.seal_containment or 100

    -- Single threshold: containment < 97% means seal has drifted outside ONE bbox
    if containment >= 97 then
        return {matched = false}
    end

    -- Build direction message from Y shift
    local direction
    if seal_y > 0 then
        direction = string.format("%.1f%% up", seal_y)
    elseif seal_y < 0 then
        direction = string.format("%.1f%% down", -seal_y)
    else
        direction = "shifted"
    end

    return {
        matched = true,
        message = string.format("Seal shift (%s, %.0f%% contained)", direction, containment)
    }
end
