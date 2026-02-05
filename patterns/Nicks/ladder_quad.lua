--[[
Pattern: LADDER_QUAD
DisplayName: Ladder & Quad
Description: 4 consecutive identical digits with remaining 4 forming a ladder
Tier: 3
Examples: ["11112345", "12345555", "43215555", "55554321"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Find quads (4 consecutive identical digits)
    for digit = 0, 9 do
        local d = tostring(digit)
        local quad = d .. d .. d .. d

        -- Check if quad exists
        local quad_start = s:find(quad, 1, true)
        if quad_start then
            -- Get the other 4 digits
            local other
            if quad_start == 1 then
                other = s:sub(5, 8)
            elseif quad_start == 5 then
                other = s:sub(1, 4)
            else
                -- Quad not at start or end
                goto continue
            end

            -- Check if other 4 form a ladder
            local is_asc = true
            local is_desc = true
            for i = 1, 3 do
                local curr = tonumber(other:sub(i, i))
                local next = tonumber(other:sub(i + 1, i + 1))
                if next - curr ~= 1 then is_asc = false end
                if curr - next ~= 1 then is_desc = false end
            end

            if is_asc or is_desc then
                local quad_positions = {}
                local ladder_positions = {}
                for i = quad_start - 1, quad_start + 2 do
                    table.insert(quad_positions, i)
                end
                if quad_start == 1 then
                    ladder_positions = {4, 5, 6, 7}
                else
                    ladder_positions = {0, 1, 2, 3}
                end

                local direction = is_asc and "ascending" or "descending"
                return {
                    matched = true,
                    message = "Quad " .. d .. "s with " .. direction .. " ladder",
                    highlights = {
                        {positions = quad_positions, color = "orange"},
                        {positions = ladder_positions, color = is_asc and "lime" or "cyan"}
                    }
                }
            end
        end
        ::continue::
    end

    return {matched = false}
end
