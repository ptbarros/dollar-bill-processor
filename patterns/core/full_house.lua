--[[
Pattern: FULL_HOUSE
Description: 5 of one digit, 3 of another
Tier: 4
Examples: ["11111222", "33333888", "55555333"]
Odds: 1 in 514
Price: $3-$8
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Require consecutive runs: five = run of exactly 5, three = run of exactly 3
    local runs = find_runs(digits)
    local five_run = nil
    local three_run = nil

    for _, run in ipairs(runs) do
        if run.length == 5 and not five_run then
            five_run = run
        elseif run.length == 3 and not three_run then
            three_run = run
        end
    end

    if not five_run or not three_run then
        return {matched = false}
    end

    local five_pos = {}
    for i = five_run.start, five_run.start + five_run.length - 1 do
        table.insert(five_pos, i)
    end

    local three_pos = {}
    for i = three_run.start, three_run.start + three_run.length - 1 do
        table.insert(three_pos, i)
    end

    return {
        matched = true,
        highlights = {
            highlight(five_pos, "gold", "five of kind"),
            highlight(three_pos, "coral", "three of kind")
        },
        connectors = {},
        message = "Full house: 5x" .. five_run.digit .. " + 3x" .. three_run.digit
    }
end
