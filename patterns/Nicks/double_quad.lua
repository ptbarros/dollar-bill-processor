--[[
Pattern: DOUBLE_QUAD
DisplayName: Double Quad
Description: First 4 digits identical, last 4 digits identical (AAAABBBB)
Tier: 2
Examples: ["11112222", "00001111", "99998888", "55556666"]
--]]

function match(ctx)
    local s = ctx.digits

    local first_digit = s:sub(1, 1)
    local last_digit = s:sub(5, 5)

    -- Check first 4 are identical
    for i = 2, 4 do
        if s:sub(i, i) ~= first_digit then
            return {matched = false}
        end
    end

    -- Check last 4 are identical
    for i = 6, 8 do
        if s:sub(i, i) ~= last_digit then
            return {matched = false}
        end
    end

    return {
        matched = true,
        message = "Double quad: " .. first_digit .. "×4 + " .. last_digit .. "×4",
        group_boxes = {
            {from = 0, to = 3, color = "orange"},
            {from = 4, to = 7, color = "cyan"}
        }
    }
end
