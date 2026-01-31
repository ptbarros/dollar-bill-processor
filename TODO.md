# Dollar Bill Processor - TODO

## Pending Features

### Middle Mouse Button Zoom (ThinkPad TrackPoint)
- **Goal**: Hold middle mouse button + move trackpoint up/down to zoom in/out
- **Status**: Not working - scroll area intercepts middle mouse events for panning
- **Attempted**:
  - Custom ZoomScrollArea with event filter on viewport
  - Overriding mouse event handlers in scroll area
  - Accepting events in PannableImageLabel to stop propagation
  - Checking for middle button in wheel events (trackpoint may generate wheel events)
- **Notes**:
  - Ctrl+wheel zoom works but is twitchy with trackpoint
  - Need to investigate Qt's internal handling of middle mouse in QScrollArea
  - May need to subclass the viewport widget itself or use a different approach

## Completed (Recent)

- [x] Fix settings checkbox naming collision (proc_auto_archive vs mon_auto_archive)
- [x] Fix Px Dev column sorting jump when viewing bills
- [x] Raise skew correction threshold from 0.2° to 1.5°
- [x] Fix view mode switching to preserve aligned images
- [x] Make Auto-Align button green when enabled
- [x] Add manual crop feature with keyboard shortcut (C)
- [x] Add Archive button for manual processing
- [x] Add multi-select support in results list
- [x] Implement lazy front/back detection
- [x] Add info icons with tooltips to settings
- [x] Make results list columns resizable with persistent widths

## Ideas / Future Enhancements

- [ ] Batch rename/export tools
- [ ] Pattern statistics dashboard
- [ ] Keyboard shortcuts reference panel
- [ ] Dark/light theme improvements
