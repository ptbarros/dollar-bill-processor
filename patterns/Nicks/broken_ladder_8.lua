--[[
Pattern: NICKS_BROKEN_LADDER_8
DisplayName: 8 Digit Broken Ladder
Description: All 8 unique digits that form consecutive sequence when sorted
Tier: 3
Examples: ["12436587", "87654312", "01325476", "98761234"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Must have 8 unique digits
    local unique = {}
    for i = 1, 8 do
        unique[s:sub(i, i)] = true
    end

    local count = 0
    for _ in pairs(unique) do count = count + 1 end

    if count ~= 8 then
        return {matched = false}
    end

    -- Sort and check if consecutive
    local digits = {}
    for i = 1, 8 do
        table.insert(digits, tonumber(s:sub(i, i)))
    end
    table.sort(digits)

    for i = 1, 7 do
        if digits[i + 1] - digits[i] ~= 1 then
            return {matched = false}
        end
    end

    -- Check it's not already a perfect ladder (sorted)
    local sorted_str = ""
    for _, d in ipairs(digits) do
        sorted_str = sorted_str .. d
    end

    if s == sorted_str then
        return {matched = false}  -- That's a regular ladder
    end

    -- Check reverse
    local reverse_sorted = ""
    for i = 8, 1, -1 do
        reverse_sorted = reverse_sorted .. digits[i]
    end
    if s == reverse_sorted then
        return {matched = false}  -- That's a regular ladder
    end

    return {
        matched = true,
        message = "Broken ladder: " .. digits[1] .. "-" .. digits[8] .. " scrambled",
        highlights = {{positions = {0,1,2,3,4,5,6,7}, color = "purple"}}
    }
end
