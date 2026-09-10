--[[
Pattern: ESS_GAS_PUMP
DisplayName: GAS_PUMP
Description: Digit vertically shifted (misaligned like old gas pump display)
Tier: 5
Examples: []
Odds: Rare - printing error
Price: $10-$100+
--]]

function match(ctx)
    -- This pattern requires metadata about baseline variance from image analysis
    local metadata = ctx.metadata
    if not metadata then
        return {matched = false}
    end

    local variance = metadata.baseline_variance
    if not variance then
        return {matched = false}
    end

    -- Use threshold from settings (slider), default to 3.5px
    local threshold = metadata.gas_pump_threshold or 3.5

    if variance < threshold then
        return {matched = false}
    end

    -- Severity tier: "strong" once the shift is >= threshold * 1.3, else "mild".
    -- Honest caveat: the mild/strong split is a soft hint -- the measurement
    -- can't reliably separate a "clear" from an "extreme" shift (the top end
    -- saturates). "gas pump vs not" (the threshold above) is the reliable line.
    -- Keep the 1.3 ratio in sync with GAS_PUMP_STRONG_RATIO in gui/results_list.py.
    local strong = metadata.gas_pump_strong_threshold or (threshold * 1.3)
    local tier = (variance >= strong) and "strong" or "mild"

    -- Highlight all digits since we don't know which one is misaligned
    local positions = {0, 1, 2, 3, 4, 5, 6, 7}

    return {
        matched = true,
        highlights = {
            highlight(positions, "red", "misaligned")
        },
        connectors = {},
        message = string.format("Gas pump error (%s, variance: %.1fpx)", tier, variance)
    }
end
