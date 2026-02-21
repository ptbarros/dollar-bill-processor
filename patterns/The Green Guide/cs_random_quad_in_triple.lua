--[[
Pattern: CS_RANDOM_QUAD_IN_TRIPLE
DisplayName: CS-Random Quad in Triple
Description: A CS-40AK (four of the same digit, scattered — no run of 4) coexists with three of a different digit and one random digit. Count distribution: {4, 3, 1}. e.g., M 34x44433 M.
BookRef: CS-270
Tier: 6
Examples: ["34744433", "12114441", "14114413"]
Odds: 1 in 22
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    local quad_digit   = nil
    local triple_digit = nil
    local single_digit = nil

    for digit, cnt in pairs(counts) do
        if cnt == 4 then
            if quad_digit then return {matched = false} end  -- only one quad allowed
            quad_digit = digit
        elseif cnt == 3 then
            if triple_digit then return {matched = false} end
            triple_digit = digit
        elseif cnt == 1 then
            if single_digit then return {matched = false} end
            single_digit = digit
        else
            return {matched = false}
        end
    end

    if not quad_digit or not triple_digit or not single_digit then
        return {matched = false}
    end

    -- Quad must be CS-40AK: no run of 4+ (scattered, not a CS-Quad)
    local runs = find_runs(d)
    for _, run in ipairs(runs) do
        if run.digit == quad_digit and run.length >= 4 then
            return {matched = false}
        end
    end

    local pos_quad   = find_digit_positions(d, quad_digit)
    local pos_triple = find_digit_positions(d, triple_digit)
    local pos_single = find_digit_positions(d, single_digit)

    -- Arc connectors linking separated quad positions
    local connectors = {}
    for i = 1, #pos_quad - 1 do
        if pos_quad[i + 1] - pos_quad[i] > 1 then
            table.insert(connectors, {
                from = pos_quad[i], to = pos_quad[i + 1],
                color = "orange", style = "arc"
            })
        end
    end

    return {
        matched = true,
        highlights = {
            {positions = pos_quad,   color = "orange"},
            {positions = pos_triple, color = "cyan"},
            {positions = pos_single, color = "coral"}
        },
        connectors = connectors,
        message = "4×" .. quad_digit .. " (scattered) + 3×" .. triple_digit .. " + 1×" .. single_digit .. " (CS-Random Quad in Triple)"
    }
end
