--[[
Pattern: CS_STAND_ALONE_LADDER
DisplayName: CS-Stand Alone Mini Ladder
Description: An ascending or descending ladder of 2+ grouped digits surrounded by zeros. e.g., M 01200000 M (2-up), M 00012300 M (3-up), M 04321000 M (4-down).
BookRef: CS-1860
Tier: 4
Examples: ["01200000", "00012300", "04321000"]
Odds: 1 in 21
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- First and last digits must be zero (surrounded)
    if d:sub(1, 1) ~= "0" or d:sub(8, 8) ~= "0" then
        return {matched = false}
    end

    -- Find the contiguous block of non-zero digits
    local start_pos, end_pos = nil, nil
    for i = 1, 8 do
        if d:sub(i, i) ~= "0" then
            if start_pos == nil then
                start_pos = i
            end
            end_pos = i
        elseif start_pos ~= nil then
            -- Check no more non-zeros after this zero
            for j = i + 1, 8 do
                if d:sub(j, j) ~= "0" then
                    return {matched = false}  -- non-contiguous non-zeros
                end
            end
            break
        end
    end

    if start_pos == nil then return {matched = false} end  -- all zeros

    local ladder_len = end_pos - start_pos + 1
    if ladder_len < 2 then return {matched = false} end  -- must be 2+ digits

    -- Check if the non-zero block forms an ascending or descending ladder
    local sub = d:sub(start_pos, end_pos)
    local ascending = is_ascending(sub)
    local descending = is_descending(sub)

    if not ascending and not descending then
        return {matched = false}
    end

    local direction = ascending and "ascending" or "descending"
    local base = start_pos - 1  -- 0-indexed

    -- Build ladder positions for highlight
    local positions = {}
    for i = base, base + ladder_len - 1 do
        table.insert(positions, i)
    end

    return {
        matched = true,
        group_boxes = {
            {from = base, to = base + ladder_len - 1, color = "lime", thickness = 3}
        },
        message = ladder_len .. "-digit " .. direction .. " ladder stand-alone (CS-Stand Alone Mini Ladder)"
    }
end
