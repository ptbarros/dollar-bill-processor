--[[
Pattern: BINARY_ROTATOR
DisplayName: Binary Rotator
Description: Turn the note upside-down and it reads the same, using only two different digits that still read upside-down — but not the plain 0-and-1 pairing (e.g. 0888·8880).
Tier: 2
Flippable: true
Odds: 1 in 2,666,667 (36 per 96M)
Examples: ["08888880", "88000088", "18888881"]
Price: $5-$100
--]]

function match(ctx)
    local d = ctx.digits

    -- Must be flip-valid and rotator
    if not all_flip_valid(d) then return {matched = false} end
    if flip_string(d) ~= d then return {matched = false} end

    -- Exactly 2 unique digits
    if unique_count(d) ~= 2 then return {matched = false} end

    -- Exclude {0,1} — that's the True Binary Rotator pattern
    local uniq = get_unique_digits(d)
    if uniq == "01" then return {matched = false} end

    -- Valid sets: {0,8} or {1,8}
    -- Note: {6,9} with 2 unique cannot form a rotator (flip maps 6->9 and 9->6,
    -- so positions would need d[k]=6 and d[9-k]=9, giving both digits present,
    -- but the rotator constraint flip_string(d)==d already handles this)

    -- Colour the two digit VALUES distinctly (each digit its own box colour),
    -- and keep the rotation arcs to show the upside-down pairing.
    local c1 = uniq:sub(1, 1)
    local highlights = {}
    for i = 0, 7 do
        local ch = d:sub(i + 1, i + 1)
        table.insert(highlights, {positions = {i}, color = (ch == c1) and "blue" or "orange"})
    end

    return {
        matched = true,
        highlights = highlights,
        connectors = {
            {from = 0, to = 7, color = "magenta", style = "arc"},
            {from = 1, to = 6, color = "magenta", style = "arc"},
        },
        message = "Binary Rotator: 2-digit rotator {" .. uniq:sub(1,1) .. "," .. uniq:sub(2,2) .. "}"
    }
end
