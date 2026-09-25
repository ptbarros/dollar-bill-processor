--[[
Pattern: TRINARY_ROTATOR
DisplayName: Trinary Rotator
Description: Turn the note upside-down and it reads the same, built from three different digits that survive the flip (e.g. 0690·0690).
Tier: 3
Flippable: true
Odds: 1 in 507,937 (189 per 96M)
Examples: ["06900690", "16911691", "10800801"]
Price: $5-$50
--]]

function match(ctx)
    local d = ctx.digits

    -- Must be flip-valid and rotator
    if not all_flip_valid(d) then return {matched = false} end
    if flip_string(d) ~= d then return {matched = false} end

    -- Exactly 3 unique digits
    if unique_count(d) ~= 3 then return {matched = false} end

    -- Colour the three digit values distinctly (by first appearance), then echo
    -- the SECOND half in the same hues but muted ("dim"), so the rotated repeat
    -- reads as a faded copy of the first half.
    local uniq = get_unique_digits(d)
    local palette = {"blue", "red", "orange"}
    local colormap = {}
    for k = 1, #uniq do
        colormap[uniq:sub(k, k)] = palette[k] or "magenta"
    end

    local highlights = {}
    for i = 0, 7 do
        local ch = d:sub(i + 1, i + 1)
        local color = colormap[ch] or "blue"
        if i < 4 then
            table.insert(highlights, {positions = {i}, color = color})
        else
            table.insert(highlights, {positions = {i}, color = color, style = "dim"})
        end
    end

    return {
        matched = true,
        highlights = highlights,
        message = "Trinary Rotator: 3-digit rotator"
    }
end
