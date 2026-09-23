| Series | \|MCS\| | Winner | T | k* | Concentration | winner_stability | Reversal | Fragile |
|---|---|---|---|---|---|---|---|---|
| PI | 1 | ar1 | 36 | 20 | 0.07 | 1.00 | no | no |
| RSAFS | 2 | seasonal_naive | 25 | 2 | 0.66 | 0.67 | no | no |
| BOPGEXP | 3 | ar1 | 30 | 2 | 0.66 | 0.74 | no | no |
| BOPGIMP | 3 | sarima | 30 | 1 | 7.70 | 0.41 | yes | yes |
| PCE | 4 | ets | 36 | 2 | 0.87 | 0.48 | yes | yes |
| UNEMPLOY | 4 | sarima | 36 | 4 | 0.92 | 0.79 | no | no |
| BUSINV | 5 | ets | 30 | 1 | 6.65 | 0.54 | no | yes |
| DGORDER | 5 | ets | 27 | 1 | 2.33 | 0.59 | no | yes |
| HOUST | 5 | theta | 36 | 1 | 1.29 | 0.22 | yes | yes |
| M2SL | 5 | ar1 | 36 | 1 | 1.30 | 0.60 | yes | yes |
| PAYEMS | 5 | ar1 | 36 | 4 | 0.37 | 0.67 | no | no |
| TCU | 5 | ar1 | 30 | 1 | 1.85 | 0.19 | yes | yes |

**Table 1.** The identification panel: all 12 curated series, sorted by |MCS| (Model Confidence Set size, α=0.10, absolute loss). 11 of 12 series have |MCS| > 1 (non-identified). k* is the decision breakdown point for the pooled winner against its most dangerous rival, over T evaluation years; read it against T, not as a bare count. Reversal is Section 5's score-vs-plurality disagreement; Fragile = winner_stability < 0.60 OR (Reversal AND k*/T < 0.25) (Section 2) -- both columns are shown so every Fragile verdict is verifiable directly from this table. All columns use equal-year weighting (w_t = 1), the convention used throughout the paper; Section 5 reports the observation-count-weighted robustness arm.
