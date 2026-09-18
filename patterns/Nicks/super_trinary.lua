--[[
Pattern: NICKS_SUPER_TRINARY
DisplayName: Super Trinary
Description: Two triples and one double, or one quad and two doubles
Tier: 4
Examples: ["11122233", "00011122", "99988877", "11112233"]
--]]

function match(ctx)
    local s = ctx.digits

    local i = 1
    local quads = 0
    local triples = 0
    local doubles = 0

    while i <= 8 do
        -- Check for quad
        if i <= 5 and s:sub(i, i) == s:sub(i + 1, i + 1) and
           s:sub(i, i) == s:sub(i + 2, i + 2) and s:sub(i, i) == s:sub(i + 3, i + 3) then
            -- Make sure it's not a quint
            if i + 4 > 8 or s:sub(i, i) ~= s:sub(i + 4, i + 4) then
                quads = quads + 1
                i = i + 4
            else
                i = i + 1
            end
        -- Check for triple
        elseif i <= 6 and s:sub(i, i) == s:sub(i + 1, i + 1) and s:sub(i, i) == s:sub(i + 2, i + 2) then
            if i + 3 > 8 or s:sub(i, i) ~= s:sub(i + 3, i + 3) then
                triples = triples + 1
                i = i + 3
            else
                i = i + 1
            end
        -- Check for double
        elseif i <= 7 and s:sub(i, i) == s:sub(i + 1, i + 1) then
            if i + 2 > 8 or s:sub(i, i) ~= s:sub(i + 2, i + 2) then
                doubles = doubles + 1
                i = i + 2
            else
                i = i + 1
            end
        else
            i = i + 1
        end
    end

    -- Two triples + one double, or one quad + two doubles
    if (triples == 2 and doubles == 1) or (quads == 1 and doubles == 2) then
        local desc = (triples == 2) and "2 triples + 1 double" or "1 quad + 2 doubles"
        -- One box around each run, no per-digit boxes (Ed review).
        local colors = {"blue", "orange", "magenta", "red"}
        local boxes, ci = {}, 1
        for _, run in ipairs(find_runs(s)) do
            if run.length >= 2 then
                table.insert(boxes, {from = run.start, to = run.start + run.length - 1,
                    color = colors[((ci - 1) % #colors) + 1], thickness = 2})
                ci = ci + 1
            end
        end
        return {
            matched = true,
            message = "Super trinary: " .. desc,
            highlights = {},
            group_boxes = boxes
        }
    end

    return {matched = false}
end
