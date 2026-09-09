# 2–3 Minute Internal Hackathon Demo

## Step 1 — Select scene
“Spectra SR starts from real Sentinel-2 L2A imagery at 10 m. For the demo I am choosing an agricultural AOI and a low-cloud acquisition.”

## Step 2 — Show source
“Here is the original 10 m observation. Fine boundaries are present only coarsely.”

## Step 3 — Run SR
“Now the SR engine reconstructs a 2.5 m product.”

## Step 4 — Slider
Drag Original ↔ Raw SR.

## Step 5 — Trust layer
“The important part is that we do not treat every generated pixel as ground truth.”
Show risk map.

## Step 6 — Safe output
Toggle Raw SR ↔ Trust-Gated SR.

Say:
“High-risk enhancement is attenuated toward a conservative source-consistent representation.”

## Step 7 — Metrics
Show the trust profile.

## Step 8 — Benchmark proof
Open benchmark page and show actual measured values from HR-reference data.

## Step 9 — Application
Show field-boundary edge comparison.

## Final line
“Spectra SR does not ask the analyst to blindly trust AI-generated detail. It reconstructs, measures reliability, and makes uncertainty visible.”
