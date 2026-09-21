--[[
Pattern: TRINARY
Description: Contains exactly 3 unique digits
Tier: 5
Examples: ["12312312", "01201201", "11223311"]
Odds: 1 in 146 (657,889 per 96M)
Price: $5-$20
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Count unique digits
    local seen = {}
    local unique = {}
    for i = 1, 8 do
        local d = digits:sub(i, i)
        if not seen[d] then
            seen[d] = true
            table.insert(unique, d)
        end
    end

    if #unique ~= 3 then
        return {matched = false}
    end

    -- Assign different colors to each unique digit
    local colors = {"blue", "orange", "magenta"}
    local digit_colors = {}
    for i, d in ipairs(unique) do
        digit_colors[d] = colors[i]
    end

    -- Highlight each digit with its assigned color, and link like occurrences
    -- with arcs in that digit's color (Ed review: add arcs back).
    local highlights = {}
    local connectors = {}
    local last = {}
    for i = 0, 7 do
        local d = digits:sub(i + 1, i + 1)
        table.insert(highlights, {
            positions = {i},
            color = digit_colors[d],
            label = "digit-" .. d
        })
        if last[d] then
            table.insert(connectors, {from = last[d], to = i, color = digit_colors[d], style = "arc"})
        end
        last[d] = i
    end

    return {
        matched = true,
        highlights = highlights,
        connectors = connectors,
        message = string.format("Trinary: digits %s, %s, %s", unique[1], unique[2], unique[3])
    }
end
