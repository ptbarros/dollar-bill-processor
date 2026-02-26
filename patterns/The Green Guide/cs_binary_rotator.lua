--[[
Pattern: CS_BINARY_ROTATOR
DisplayName: CS-Binary Rotator
Description: Rotator using exactly 2 unique digits from {0,8} or {1,8}. Excludes {0,1} which is CS-True Binary Rotator (CS-1110).
BookRef: CS-1120
Tier: 3
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

    -- Exclude {0,1} — that's CS-1110 True Binary Rotator
    local uniq = get_unique_digits(d)
    if uniq == "01" then return {matched = false} end

    -- Valid sets: {0,8} or {1,8}
    -- Note: {6,9} with 2 unique cannot form a rotator (flip maps 6->9 and 9->6,
    -- so positions would need d[k]=6 and d[9-k]=9, giving both digits present,
    -- but the rotator constraint flip_string(d)==d already handles this)

    local positions = {0, 1, 2, 3, 4, 5, 6, 7}

    return {
        matched = true,
        highlights = {{positions = positions, color = "purple"}},
        connectors = {
            {from = 0, to = 7, color = "purple", style = "arc"},
            {from = 1, to = 6, color = "purple", style = "arc"},
        },
        message = "CS-Binary Rotator: 2-digit rotator {" .. uniq:sub(1,1) .. "," .. uniq:sub(2,2) .. "} (CS-1120)"
    }
end
