--[[
Pattern: TRIPLE_DOUBLE
Description: One triple and two doubles as consecutive runs (AAABBBCC or AABBCCC format)
Tier: 4
Examples: ["11122233", "44455566", "11233344"]
Odds: 1 in ~5,000
Price: $25-$75
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Find consecutive runs
    local runs = {}
    local i = 1
    while i <= 8 do
        local d = digits:sub(i, i)
        local run_start = i
        local run_len = 1
        while i + run_len <= 8 and digits:sub(i + run_len, i + run_len) == d do
            run_len = run_len + 1
        end
        table.insert(runs, {digit = d, start = run_start - 1, length = run_len})
        i = i + run_len
    end

    -- Sort run lengths to check pattern
    local lengths = {}
    for _, r in ipairs(runs) do
        table.insert(lengths, r.length)
    end
    table.sort(lengths, function(a, b) return a > b end)

    -- Check for exactly: one triple (3), two doubles (2, 2), one single (1)
    local is_triple_double = (
        #lengths == 4 and
        lengths[1] == 3 and
        lengths[2] == 2 and
        lengths[3] == 2 and
        lengths[4] == 1
    )

    if not is_triple_double then
        return {matched = false}
    end

    -- Build highlights for each run
    local colors = {"gold", "orange", "coral", "salmon"}
    local highlights = {}
    local connectors = {}

    for idx, r in ipairs(runs) do
        local color = colors[math.min(idx, 4)]
        local positions = {}
        for p = r.start, r.start + r.length - 1 do
            table.insert(positions, p)
        end

        local label = "single"
        if r.length == 2 then label = "double"
        elseif r.length == 3 then label = "triple"
        end

        table.insert(highlights, {
            positions = positions,
            color = color,
            label = label
        })

        -- Connect runs of 2+
        if r.length >= 2 then
            table.insert(connectors, {
                from = r.start,
                to = r.start + r.length - 1,
                color = color,
                style = "bracket"
            })
        end
    end

    return {
        matched = true,
        highlights = highlights,
        connectors = connectors,
        message = "Triple + double + double pattern"
    }
end
