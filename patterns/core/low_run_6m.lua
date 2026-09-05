--[[
Pattern: LOW_RUN_6M
DisplayName: Low Run 6.4M
Description: Match bills from 6.4 million BEP print runs
Tier: 5
DataFile: low_runs.csv
Examples: []
Odds: Varies by series/district
Price: $5-$25
--]]

function match(ctx)
    if not ctx.data then return {matched = false} end

    -- Need series_year from plate extraction
    local series = ctx.metadata and ctx.metadata.series_year or nil
    if not series or series == "" then return {matched = false} end

    -- Normalize series (strip "SERIES " prefix if OCR included it)
    series = string.gsub(series, "^SERIES%s*", "")

    -- District = first letter of serial, Block = last letter
    local district = string.sub(ctx.full_serial, 1, 1)
    local block = string.sub(ctx.full_serial, #ctx.full_serial, #ctx.full_serial)

    -- Facility from front_plate (FW prefix = Fort Worth, else DC)
    local front_plate_raw = ctx.metadata and ctx.metadata.front_plate or ""
    local facility = ""
    if starts_with(front_plate_raw, "FW") then
        facility = "FW"
    elseif front_plate_raw ~= "" then
        facility = "DC"
    end

    -- Look up in low runs data - only 6.4M runs
    for _, row in ipairs(ctx.data) do
        if row.series == series
           and row.district == district
           and row.block == block
           and row.facility == facility
           and row.quantity == "6.4" then
            return {
                matched = true,
                message = "Low run (6.4M): Series " .. series
                    .. " District " .. district
                    .. " Block " .. block
                    .. " " .. facility
            }
        end
    end

    return {matched = false}
end
