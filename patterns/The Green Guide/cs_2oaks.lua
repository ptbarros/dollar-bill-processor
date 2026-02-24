--[[
Pattern: CS_2OAKS
DisplayName: CS-2OAKs
Description: Any two of the same digit anywhere in the serial, as long as the two digits are not grouped (not adjacent). A CS-2OAK is two non-consecutive matching digits. e.g., M xx2x2xxx M or M 2xx2xxxx M.
BookRef: CS-10
Tier: 9
Examples: ["12032145", "20103456", "10234512"]
Odds: 1 in many
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    -- Find digits appearing exactly twice
    local oak_digits = {}
    for digit, cnt in pairs(counts) do
        if cnt == 2 then
            local positions = find_digit_positions(d, digit)
            -- Must be non-adjacent (not a pair)
            if positions[2] - positions[1] > 1 then
                table.insert(oak_digits, digit)
            end
        end
    end

    -- Must have at least one 2OAK (non-grouped pair)
    if #oak_digits == 0 then
        return {matched = false}
    end

    -- Sort for consistent output
    table.sort(oak_digits)

    local colors = {"orange", "coral", "cyan", "lime", "purple", "teal"}
    local highlights = {}
    local connectors = {}

    for i, digit in ipairs(oak_digits) do
        local positions = find_digit_positions(d, digit)
        local color = colors[((i - 1) % #colors) + 1]
        table.insert(highlights, {positions = positions, color = color})
        table.insert(connectors, {from = positions[1], to = positions[2], color = color, style = "arc"})
    end

    local msg = #oak_digits .. " pair(s) of scattered digits (CS-2OAKs)"
    return {
        matched = true,
        highlights = highlights,
        connectors = connectors,
        message = msg
    }
end
