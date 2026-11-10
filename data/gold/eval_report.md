# Comment tagging — accuracy against the gold set

Generated 2026-11-10 by tools/eval_tagging.py · taxonomy 2026-11-rules-v2. Gold set: 300 labelled comments (0 verified by a person, 300 draft). Convention: a complaint or an unmet-need request is Negative. Draft labels were pre-filled by an AI assistant and are NOT a validation — only verified rows count as evidence of accuracy.

## Source: rules

### Verified labels

No labels in this group yet.

### Draft labels (unverified)

Comments scored: 300. Themes — micro precision 69%, recall 67%, F1 68%; macro F1 63%.
Sentiment accuracy 63% on 300 comments; of the errors, 109 were more positive than the label and 2 more negative.

| Theme | Labelled | Predicted | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Space Availability & Crowding | 61 | 12 | 83% | 16% | 27% |
| Hardware (Monitors / Peripherals) | 59 | 49 | 86% | 71% | 78% |
| Privacy & Confidentiality | 54 | 54 | 76% | 76% | 76% |
| Noise & Distraction | 48 | 50 | 82% | 85% | 84% |
| Desk Availability & Reservations | 40 | 37 | 76% | 70% | 73% |
| Positive / Appreciation | 35 | 34 | 62% | 60% | 61% |
| In-Person / Team Presence | 32 | 13 | 92% | 38% | 53% |
| RTO / Hybrid Policy | 32 | 22 | 68% | 47% | 56% |
| Food & Beverage | 31 | 38 | 76% | 94% | 84% |
| Temperature / Thermal | 28 | 26 | 92% | 86% | 89% |
| Connectivity (WiFi / Login / Power) | 27 | 35 | 71% | 93% | 81% |
| Focus / Concentration | 23 | 32 | 69% | 96% | 80% |
| Cleanliness & Upkeep | 20 | 19 | 84% | 80% | 82% |
| Meeting & Conference Rooms | 18 | 23 | 52% | 67% | 59% |
| Parking & Commute | 18 | 17 | 88% | 83% | 86% |
| AV / Video Conferencing | 16 | 11 | 91% | 62% | 74% |
| Support / Help Services | 15 | 3 | 100% | 20% | 33% |
| Furniture & Ergonomics | 14 | 11 | 91% | 71% | 80% |
| Unclassified | 11 | 38 | 26% | 91% | 41% |
| Cost / Pricing | 11 | 9 | 78% | 64% | 70% |
| Lighting | 9 | 13 | 62% | 89% | 73% |
| Building Operations | 9 | 14 | 57% | 89% | 70% |
| Software / Tools | 7 | 6 | 50% | 43% | 46% |
| Community & Events | 7 | 14 | 50% | 100% | 67% |
| Reception & Entry | 7 | 10 | 60% | 86% | 71% |
| Wellbeing / Wellness Spaces | 4 | 2 | 50% | 25% | 33% |
| Communication & Updates | 4 | 9 | 22% | 50% | 31% |
| Navigation / Wayfinding | 2 | 1 | 100% | 50% | 67% |
| Overall Dissatisfaction | 2 | 14 | 7% | 50% | 12% |
| Collaboration Space | 0 | 4 | 0% | — | 0% |

Sentiment confusion (rows = label, columns = predicted):

| label \ predicted | Negative | Neutral / Mixed | Positive |
|---|---:|---:|---:|
| Negative | 161 | 63 | 32 |
| Neutral / Mixed | 2 | 9 | 14 |
| Positive | 0 | 0 | 19 |
