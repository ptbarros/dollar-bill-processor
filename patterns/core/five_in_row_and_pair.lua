--[[
Pattern: FIVE_IN_ROW_AND_PAIR
DisplayName: 5 in a Row & Pair
Description: Exactly five of a kind in a row plus a single separate pair (e.g. 11111·22·3).
Tier: 5
Examples: ["11111223", "55555667", "99999100"]
Odds: 1 in 11,134 (8,622 per 96M)
Price: $10-$50
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Find a run of 5.
    local five_run = has_n_consecutive(digits, 5)
    if not five_run then
        return {matched = false}
    end

    -- The run must be EXACTLY five (Ed review): reject if the same digit extends
    -- immediately before or after the run (that would be six-or-more in a row).
    local rs = five_run.start  -- 0-indexed
    if (rs - 1 >= 0 and digits:sub(rs, rs) == five_run.digit) or
       (rs + 5 <= 7 and digits:sub(rs + 6, rs + 6) == five_run.digit) then
        return {matched = false}
    end

    -- Exactly one other digit forms a pair (count == 2). A triple/quad of another
    -- digit does NOT count as a pair (Ed review).
    local counts = count_digits(digits)
    local pair_digit = nil
    for d, c in pairs(counts) do
        if d ~= five_run.digit and c == 2 then
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
