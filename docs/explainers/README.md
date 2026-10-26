# Explainers

Visual explainers for how Dollar Detective's detection columns work. Built from
real scanned bills (registered + zoomed on the detected region, with the live
metric values overlaid).

## Seal Shift

Shows the green Treasury seal drifting off the engraved "ONE" as the overprint
plate registration changes. The "ONE" box is pinned in place so the seal visibly
moves; the red area is the part of the seal that has escaped the box (what the
**Containment** column measures), and the amber arrow is the shift vector.

| File | Use |
|------|-----|
| `seal-shift.html` | Full interactive page — play/scrub both sweeps, toggle Seal&nbsp;Y vs Seal&nbsp;X, light/dark. Self-contained (frames embedded). Best for the docs site. |
| `media/seal-shift-vertical.mp4` / `.gif` | Seal&nbsp;Y sweep, +4% → 0 → −6% (ping-pong loop). |
| `media/seal-shift-horizontal.mp4` / `.gif` | Seal&nbsp;X sweep, +4% → −9% (ping-pong loop). |

### Embedding

On the docs site (GitHub Pages — HTML works), prefer the MP4 or link the
interactive page:

```html
<video controls loop muted playsinline width="720">
  <source src="media/seal-shift-vertical.mp4" type="video/mp4">
</video>
<p><a href="seal-shift.html">Open the interactive version →</a></p>
```

In a Markdown README on github.com (scripts/iframes are stripped, so the
interactive page won't run there) use the GIF:

```markdown
![Seal shift — vertical](docs/explainers/media/seal-shift-vertical.gif)
```

Regenerate with the scripts in `tools/explainers/` (re-runs across the current
straps and rebuilds these files).
