--[[
Pattern: BROKEN_LADDER_8
Description: 8-digit broken ladder (one digit out of sequence)
Tier: 4
Examples: ["23546718", "12354678"]
Odds: 1 in 2,380
Price: $10-$35
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check if digits would form a ladder with exactly one swap
    -- Get all 8 digits
    local nums = {}
    for i = 1, 8 do
        table.insert(nums, tonumber(digits:sub(i, i)))
    end

    -- Check ascending with one error
    local asc_errors = 0
    local asc_error_pos = nil
    for i = 1, 7 do
        if nums[i + 1] ~= nums[i] + 1 then
            asc_errors = asc_errors + 1
            asc_error_pos = i
        end
    end

    -- Check descending with one error
    local desc_errors = 0
    local desc_error_pos = nil
    for i = 1, 7 do
        if nums[i + 1] ~= nums[i] - 1 then
            desc_errors = desc_errors + 1
            desc_error_pos = i
        end
    end

    if asc_errors ~= 1 and desc_errors ~= 1 then
        return {matched = false}
    end

    local error_pos = asc_errors == 1 and asc_error_pos or desc_error_pos
    local direction = asc_errors == 1 and "ascending" or "descending"

    -- Highlight ladder positions in lime, broken position in red
    local positions = {}
    for i = 0, 7 do
        table.insert(positions, i)
    end

    return {
        matched = true,
        highlights = {
            highlight(positions, "lime", "broken ladder"),
            highlight({error_pos - 1, error_pos}, "red", "break")
        },
        connectors = {
            connector(error_pos - 1, error_pos, "red", "dashed")
        },
        message = "Broken " .. direction .. " ladder"
    }
end
