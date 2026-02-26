--[[
Pattern: CS_REVERSE_DATE_NOTES
DisplayName: CS-Reverse Date Notes
Description: The serial, when reversed, forms a valid calendar date in US (mmddyyyy), EU (ddmmyyyy), or INTL (yyyymmdd) format.
BookRef: CS-510
Tier: 9
Examples: ["57910221", "57910122", "58915110"]
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    local rev = string.reverse(d)

    -- Try US format: mmddyyyy
    local us_mm = tonumber(rev:sub(1, 2))
    local us_dd = tonumber(rev:sub(3, 4))
    local us_yyyy = tonumber(rev:sub(5, 8))
    if is_valid_date(us_mm, us_dd, us_yyyy) then
        return {
            matched = true,
            highlights = {
                highlight_range(0, 7, "purple")
            },
            connectors = {
                {from = 0, to = 7, color = "coral", style = "arc"}
            },
            message = string.format("Reverse Date: %s → %s = %02d/%02d/%04d (US)", d, rev, us_mm, us_dd, us_yyyy)
        }
    end

    -- Try EU format: ddmmyyyy
    local eu_dd = tonumber(rev:sub(1, 2))
    local eu_mm = tonumber(rev:sub(3, 4))
    local eu_yyyy = tonumber(rev:sub(5, 8))
    if is_valid_date(eu_mm, eu_dd, eu_yyyy) then
        return {
            matched = true,
            highlights = {
                highlight_range(0, 7, "purple")
            },
            connectors = {
                {from = 0, to = 7, color = "coral", style = "arc"}
            },
            message = string.format("Reverse Date: %s → %s = %02d/%02d/%04d (EU)", d, rev, eu_dd, eu_mm, eu_yyyy)
        }
    end

    -- Try INTL format: yyyymmdd
    local intl_yyyy = tonumber(rev:sub(1, 4))
    local intl_mm = tonumber(rev:sub(5, 6))
    local intl_dd = tonumber(rev:sub(7, 8))
    if is_valid_date(intl_mm, intl_dd, intl_yyyy) then
        return {
            matched = true,
            highlights = {
                highlight_range(0, 7, "purple")
            },
            connectors = {
                {from = 0, to = 7, color = "coral", style = "arc"}
            },
            message = string.format("Reverse Date: %s → %s = %04d-%02d-%02d (INTL)", d, rev, intl_yyyy, intl_mm, intl_dd)
        }
    end

    return {matched = false}
end
