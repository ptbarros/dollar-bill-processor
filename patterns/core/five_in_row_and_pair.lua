--[[
Pattern: FIVE_IN_ROW_AND_PAIR
Description: 5 in a row plus a pair
Tier: 3
Examples: ["11111122", "33333344", "22255555"]
Odds: 1 in 66,667
Price: $10-$50
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Find a run of 5+
    local five_run = has_n_consecutive(digits, 5)
    if not five_run then
        return {matched = false}
    end

    -- Find a pair among remaining digits
    local counts = count_digits(digits)
    local pair_digit = nil
    for d, c in pairs(counts) do
        if d ~= five_run.digit and c >= 2 then
            pair_digit = d
            break
        end
    end

    if not pair_digit then
        return {matched = false}
    end

    local run_pos = {}
    for i = 0, 4 do
        table.insert(run_pos, five_run.start + i)
    end

    local pair_pos = find_digit_positions(digits, pair_digit)

    return {
        matched = true,
        highlights = {
            highlight(run_pos, "gold", "5 in row"),
            highlight(pair_pos, "coral", "pair")
        },
        connectors = {},
        message = "5 x " .. five_run.digit .. " + pair of " .. pair_digit .. "s"
    }
end
