# Answer key: analysis (planted in the generator, make_tasks.py)
Planted effects (clean data, before the planted errors):
1. Weekend effect: Saturday/Sunday demand is 1.6x weekdays (avg daily revenue weekend $4,477 vs weekday $2,820, ratio 1.59).
2. "Seasonal galette" declines steadily (demand -60% from the first day to the last; monthly units Jan 992 -> Jun 542, -45%); every other product is flat.
3. Store Elm jumps from 2026-03-16 (avg daily revenue 1,211 before vs 1,603 after, +32%; demand +35%); Maple and Oak are flat.
   Store sizes differ: Elm > Maple (0.85x) > Oak (0.70x).
Planted data problems:
- 24 exact duplicate rows; - 6 rows with negative quantities; - 3 rows with an empty unit_price.
A strong report finds all three effects with numbers, notices all three data problems and says how it handled them,
and makes actionable recommendations (e.g. weekend staffing/stock, review or drop the galette, find out what changed at
Elm on Mar 16 and copy it). Claims of effects that are not in the data (e.g. a product trend other than the galette)
are errors.
