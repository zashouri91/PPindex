# PPIndex — changes from the November 2026 review

Branch `fix/review-2026-11`, based on `45dba0b`. Population for every figure below: completed responses at the 10 in-scope sites unless stated. This document contains no comment text, respondent IDs, Domain IDs or IP addresses.

Verify locally: `pip install -r requirements.txt -r requirements-dev.txt`, then `pytest` and `python build_data.py` (reads `data/raw_deid.xlsx`).

## 1. Commits

| Commit | Change |
|---|---|
| `c53fab2` | Phase 0: make report builders importable without Streamlit; add baseline tests |
| `6ab5748` | Phase 1: track a de-identified raw export; stop tracking the original |
| `2df2f70` | Phase 2.1: "What's working" prints the true positive share |
| `f9b0509` | Phase 2.2: one GENERIC_THEMES set, excluded from every priority ranking |
| `ee37937` | Phase 2.3: replace "represents ~N similar comments" |
| `c21256f` | Phase 2.4: make the sentiment question prior fire |
| `213e4c1` | Phase 2.5: action-plan payload sends the site's respondent count |
| `225ace0` | Phase 2.6: one source for item-level status |
| `e05e41e` | Phase 2.7: copy fixes |
| `672fe04` | Phase 3.1: 95% intervals for site Index/dimension scores and item gaps |
| `0b48ac4` | Phase 3.2: standing replaces rank numbers and thirds |
| `5fc958d` | Phase 3.3: one definition of "company" |
| `9f6f643` | Phase 3.4: sentiment-item table uses margin + standing, not Welch p-values |
| `2894bca` | Phase 3.5: key drivers compare commenters with other commenters |
| `9340b7b` | Phase 3.6: theme prevalence counts people, not comments |
| `6692ae9` | Phase 3.7: severity words alongside the official status |
| `9d1cd23` | Phase 4A: tighten the keyword rules in taxonomy.py; rebuild comments |
| `9724383` | Phase 4B: gold set, labelling view and accuracy report |
| `a73f126` | Phase 4C: shared model backends; optional local-model tagger (off) |
| `6a2a34e` | Phase 4D: honest accuracy note; rebuild check |
| `cdc9cd6` | Phase 5: site card content and layout |
| `ada69b3` | Phase 6: portfolio card |
| `43d8eba` | Phase 7: one colour system, contrast and size tests |
| `b509c23` | Phase 8: action plans — payload, prompt v3.2, grounding check, stored evidence |
| `1b604c3` | Phase 9: README and samples |

## 2. Section 4 baseline — unchanged

| Site | Brief | Now | Δ |
|---|---:|---:|---:|
| GA-Atlanta | 76.54 | 76.541 | +0.001 |
| FL-Miami | 71.70 | 71.702 | +0.002 |
| NY-New York (Penn1) | 71.35 | 71.350 | +0.000 |
| IL-Chicago-Willis Tower | 70.24 | 70.238 | -0.002 |
| VA-Norfolk | 69.78 | 69.775 | -0.005 |
| OH-Mason | 68.44 | 68.442 | +0.002 |
| IN-Indianapolis | 68.17 | 68.168 | -0.002 |
| FL-Tampa | 67.76 | 67.755 | -0.005 |
| VA-Richmond | 66.54 | 66.540 | +0.000 |
| CA-Woodland Hills | 65.75 | 65.750 | +0.000 |

Company Index 69.0; Work Activity 62.5, Physical Feature 66.1, Service Level 75.6, Digital & Tech 72.0. Status across the 235 site-by-item cells with an Importance rating: 76 Under Supported, 148 Target, 11 Over Supported. All held after every phase (`tests/test_baseline.py`, ±0.01 on the site Index). Tampa computes to 67.754, which the brief rounds to 67.76; inside the tolerance.

## 3. Review baselines — reproduction

- Reproduced: Woodland Hills Reception & Entry printed 91% positive where the positive share is 36%.
- Reproduced: Policy / Experience in the top five at 9 of 10 sites, 55–73% of its comments praise tags (now 3 of 10, on RTO / Hybrid comments).
- Reproduced: Indianapolis top quote claimed ~177 similar comments.
- Reproduced: one status cell differs between computations (New York, Planned in-person meetings, gap +1.515).
- Reproduced: commenters 3.57 vs non-commenters 4.08 (959 completers wrote nothing); 28 of 29 themes looked like drags before generic buckets were excluded.
- Reproduced: 23% of commenters wrote 41% of comments (23.4% / 41.0%: people with 4–5 comments).
- Reproduced: 12 of the 76 red cells are borderline; 13 Target cells have Support under 60.
- Reproduced: 94 comments carry a role-and-tenure combination shared by fewer than five respondents at the site; 20 by one person.
- Reproduced: 5,157 completed comments (all sites); Desk Availability 1,459, AV 353 (272 on bare “teams”), Collaboration Space 381 (363 on bare “collaborat”); Unclassified 9.4%.
- **Not reproduced as stated:** “814 of 1,459 Desk comments rest only on desk / workstation / cubicle / cube / seat”. 814 reproduces when “seat” is left out of that set; with “seat” it is 897.
- Reproduced: the sentiment prior fix changes up to ~8% of labels (7.8% of all non-junk; 7.7% of the card population), all toward negative, on two questions.
- Not reproducible here: “60 comments read blind, shipped label matched 41” — the 60 IDs are not in the brief. On the 300-comment draft gold set the shipped labels at 45dba0b match 60%, and 119 of 121 misses are more positive than the comment, consistent with the review.
- Reproduced: 11 comments from 9 people at Indianapolis mention the two-week booking window (“about a dozen”).

## 4. Per site — before → after

Before = the original code at `45dba0b`. After = this branch. Themes after are by people (share of the site's respondents); before they were comment counts with the old keyword rules.

### CA-Woodland Hills

- **Standing.** Before: rank 10 of 10, “Bottom third”. After: Index 66 ± 2, **Below the company average** (company 69.0); dimensions — Work Activity 56 ± 3 Below; Physical Feature 64 ± 2 Below; Service Level 73 ± 2 Below; Digital & Tech 71 ± 2 In line.
- **Top themes.** Before: Desk Availability & Reservations (174 comments); Noise & Distraction (137 comments); Privacy & Confidentiality (100 comments); Hardware (Monitors / Peripherals) (98 comments); Food & Beverage (86 comments).
  After: Noise & Distraction (105 people, 27%); Privacy & Confidentiality (75 people, 19%); Hardware (Monitors / Peripherals) (74 people, 19%); Food & Beverage (72 people, 18%); Desk Availability & Reservations (66 people, 17%).
- **What's working.** Before: Parking & Commute (70% “positive”); Cleanliness & Upkeep (72% “positive”); Community & Events (84% “positive”); Reception & Entry (91% “positive”); Support / Help Services (74% “positive”). After: Wellbeing / Wellness Spaces (57% positive, 13 people); Entry and reception (rating, support 81).
- **Red items by severity** (7; same set as before — status is unchanged): severe: Focused individual work that requires concentration and minimal interruption, Availability of spaces that support focused work · moderate: Availability of spaces that support discretion, Tasks requiring discretion · slight: Virtual collaboration with remote or hybrid colleagues, Thermal / temperature comfort, Furniture comfort and ergonomics.

### FL-Miami

- **Standing.** Before: rank 2 of 10, “Top third”. After: Index 72 ± 4, **In line with the company average** (company 69.0); dimensions — Work Activity 66 ± 6 In line; Physical Feature 70 ± 4 In line; Service Level 73 ± 3 In line; Digital & Tech 77 ± 4 Above.
- **Top themes.** Before: Desk Availability & Reservations (26 comments); Noise & Distraction (24 comments); Hardware (Monitors / Peripherals) (24 comments); Focus / Concentration (20 comments); Connectivity (WiFi / Login / Power) (18 comments).
  After: Noise & Distraction (17 people, 17%); Hardware (Monitors / Peripherals) (16 people, 16%); Focus / Concentration (15 people, 15%); Connectivity (WiFi / Login / Power) (12 people, 12%); Temperature / Thermal (12 people, 12%).
- **What's working.** Before: Desk Availability & Reservations (73% “positive”); Connectivity (WiFi / Login / Power) (72% “positive”); Reception & Entry (70% “positive”); Software / Tools (86% “positive”); In-Person / Team Presence (80% “positive”). After: Cleanliness and upkeep of PulsePoint (rating, support 81); Entry and reception (rating, support 80).
- **Red items by severity** (10; same set as before — status is unchanged): severe: Focused individual work that requires concentration and minimal interruption · moderate: Thermal / temperature comfort, Availability of spaces that support focused work, Tasks requiring discretion, Availability of spaces that support discretion · slight: Virtual collaboration with remote or hybrid colleagues, Furniture comfort and ergonomics (borderline), Learning, training, or skill development (borderline), Lighting comfort (borderline), Reliability of building operations (borderline).

### FL-Tampa

- **Standing.** Before: rank 8 of 10, “Bottom third”. After: Index 68 ± 2, **In line with the company average** (company 69.0); dimensions — Work Activity 60 ± 4 In line; Physical Feature 67 ± 3 In line; Service Level 74 ± 2 In line; Digital & Tech 70 ± 3 In line.
- **Top themes.** Before: Desk Availability & Reservations (77 comments); Hardware (Monitors / Peripherals) (64 comments); Privacy & Confidentiality (57 comments); Noise & Distraction (47 comments); Food & Beverage (42 comments).
  After: Hardware (Monitors / Peripherals) (47 people, 27%); Privacy & Confidentiality (39 people, 22%); Noise & Distraction (38 people, 21%); Food & Beverage (34 people, 19%); Desk Availability & Reservations (29 people, 16%).
- **What's working.** Before: Cleanliness & Upkeep (88% “positive”); Reception & Entry (73% “positive”); Wellbeing / Wellness Spaces (83% “positive”). After: Cleanliness & Upkeep (62% positive, 20 people); Cleanliness and upkeep of PulsePoint (rating, support 83).
- **Red items by severity** (6; same set as before — status is unchanged): severe: Focused individual work that requires concentration and minimal interruption · moderate: Availability of spaces that support focused work, Availability of spaces that support discretion, Tasks requiring discretion, Thermal / temperature comfort · slight: Virtual collaboration with remote or hybrid colleagues.

### GA-Atlanta

- **Standing.** Before: rank 1 of 10, “Top third”. After: Index 77 ± 2, **Above the company average** (company 69.0); dimensions — Work Activity 71 ± 3 Above; Physical Feature 74 ± 2 Above; Service Level 83 ± 2 Above; Digital & Tech 78 ± 2 Above.
- **Top themes.** Before: Desk Availability & Reservations (98 comments); Noise & Distraction (58 comments); Privacy & Confidentiality (57 comments); Hardware (Monitors / Peripherals) (55 comments); Focus / Concentration (51 comments).
  After: Noise & Distraction (41 people, 15%); Privacy & Confidentiality (41 people, 15%); Food & Beverage (41 people, 15%); Hardware (Monitors / Peripherals) (39 people, 14%); Desk Availability & Reservations (39 people, 14%).
- **What's working.** Before: Parking & Commute (73% “positive”); AV / Video Conferencing (70% “positive”); Collaboration Space (72% “positive”); Building Operations (74% “positive”); Cleanliness & Upkeep (73% “positive”). After: Cleanliness & Upkeep (59% positive, 20 people); Reception & Entry (82% positive, 15 people); RTO / Hybrid Policy (50% positive, 13 people); Building Operations (53% positive, 13 people).
- **Red items by severity** (5; same set as before — status is unchanged): moderate: Availability of spaces that support focused work, Focused individual work that requires concentration and minimal interruption · slight: Availability of spaces that support discretion, Thermal / temperature comfort, Virtual collaboration with remote or hybrid colleagues (borderline).

### IL-Chicago-Willis Tower

- **Standing.** Before: rank 4 of 10, “Top third”. After: Index 70 ± 3, **In line with the company average** (company 69.0); dimensions — Work Activity 60 ± 4 In line; Physical Feature 68 ± 3 In line; Service Level 81 ± 3 Above; Digital & Tech 71 ± 3 In line.
- **Top themes.** Before: Desk Availability & Reservations (45 comments); Hardware (Monitors / Peripherals) (39 comments); Privacy & Confidentiality (32 comments); Noise & Distraction (27 comments); Temperature / Thermal (17 comments).
  After: Hardware (Monitors / Peripherals) (23 people, 26%); Privacy & Confidentiality (19 people, 22%); Noise & Distraction (19 people, 22%); Desk Availability & Reservations (19 people, 22%); Temperature / Thermal (14 people, 16%).
- **What's working.** Before: Community & Events (100% “positive”). After: Support from a community manager (rating, support 90); Entry and reception (rating, support 89).
- **Red items by severity** (7; same set as before — status is unchanged): severe: Focused individual work that requires concentration and minimal interruption · moderate: Availability of spaces that support discretion, Availability of spaces that support focused work, Virtual collaboration with remote or hybrid colleagues, Tasks requiring discretion · slight: Thermal / temperature comfort (borderline), Availability of on-site food and beverage options (borderline).

### IN-Indianapolis

- **Standing.** Before: rank 7 of 10, “Bottom third”. After: Index 68 ± 1, **In line with the company average** (company 69.0); dimensions — Work Activity 63 ± 2 In line; Physical Feature 63 ± 1 Below; Service Level 77 ± 1 Above; Digital & Tech 70 ± 1 Below.
- **Top themes.** Before: Desk Availability & Reservations (375 comments); Hardware (Monitors / Peripherals) (237 comments); Privacy & Confidentiality (224 comments); Noise & Distraction (205 comments); Connectivity (WiFi / Login / Power) (167 comments).
  After: Hardware (Monitors / Peripherals) (176 people, 26%); Privacy & Confidentiality (162 people, 23%); Noise & Distraction (160 people, 23%); Desk Availability & Reservations (147 people, 21%); Connectivity (WiFi / Login / Power) (131 people, 19%).
- **What's working.** Before: Wellbeing / Wellness Spaces (71% “positive”); Reception & Entry (81% “positive”); Community & Events (80% “positive”); In-Person / Team Presence (72% “positive”); Support / Help Services (79% “positive”). After: Reception & Entry (53% positive, 42 people); Wellbeing / Wellness Spaces (55% positive, 20 people).
- **Red items by severity** (7; same set as before — status is unchanged): moderate: Availability of spaces that support discretion, Focused individual work that requires concentration and minimal interruption, Availability of spaces that support focused work, Thermal / temperature comfort, Tasks requiring discretion · slight: Virtual collaboration with remote or hybrid colleagues, Furniture comfort and ergonomics.

### NY-New York (Penn1)

- **Standing.** Before: rank 3 of 10, “Top third”. After: Index 71 ± 3, **In line with the company average** (company 69.0); dimensions — Work Activity 65 ± 5 In line; Physical Feature 71 ± 4 Above; Service Level 78 ± 3 In line; Digital & Tech 72 ± 4 In line.
- **Top themes.** Before: Desk Availability & Reservations (40 comments); Privacy & Confidentiality (30 comments); Noise & Distraction (28 comments); Connectivity (WiFi / Login / Power) (27 comments); Hardware (Monitors / Peripherals) (26 comments).
  After: Privacy & Confidentiality (22 people, 22%); Noise & Distraction (22 people, 22%); Temperature / Thermal (20 people, 20%); Hardware (Monitors / Peripherals) (19 people, 19%); Focus / Concentration (19 people, 19%).
- **What's working.** Before: Community & Events (100% “positive”); Parking & Commute (100% “positive”). After: Entry and reception (rating, support 85); Cleanliness and upkeep of PulsePoint (rating, support 85).
- **Red items by severity** (7; same set as before — status is unchanged): moderate: Focused individual work that requires concentration and minimal interruption, Availability of spaces that support discretion, Availability of spaces that support focused work, Thermal / temperature comfort · slight: Tasks requiring discretion, Virtual collaboration with remote or hybrid colleagues, Availability of on-site food and beverage options (borderline).

### OH-Mason

- **Standing.** Before: rank 6 of 10, “Middle third”. After: Index 68 ± 2, **In line with the company average** (company 69.0); dimensions — Work Activity 61 ± 2 In line; Physical Feature 66 ± 2 In line; Service Level 76 ± 2 In line; Digital & Tech 71 ± 2 In line.
- **Top themes.** Before: Desk Availability & Reservations (194 comments); Noise & Distraction (137 comments); Privacy & Confidentiality (124 comments); Hardware (Monitors / Peripherals) (115 comments); Focus / Concentration (96 comments).
  After: Noise & Distraction (105 people, 29%); Privacy & Confidentiality (90 people, 25%); Hardware (Monitors / Peripherals) (82 people, 23%); Desk Availability & Reservations (82 people, 23%); Food & Beverage (72 people, 20%).
- **What's working.** Before: Community & Events (83% “positive”); Reception & Entry (77% “positive”); Support / Help Services (85% “positive”). After: Reception & Entry (63% positive, 19 people); Support / Help Services (54% positive, 12 people); Wellbeing / Wellness Spaces (55% positive, 10 people).
- **Red items by severity** (6; same set as before — status is unchanged): severe: Availability of spaces that support focused work, Focused individual work that requires concentration and minimal interruption · moderate: Availability of spaces that support discretion, Tasks requiring discretion · slight: Virtual collaboration with remote or hybrid colleagues, Thermal / temperature comfort.

### VA-Norfolk

- **Standing.** Before: rank 5 of 10, “Middle third”. After: Index 70 ± 1, **In line with the company average** (company 69.0); dimensions — Work Activity 64 ± 2 In line; Physical Feature 66 ± 2 In line; Service Level 75 ± 2 In line; Digital & Tech 74 ± 2 Above.
- **Top themes.** Before: Desk Availability & Reservations (222 comments); Hardware (Monitors / Peripherals) (150 comments); Privacy & Confidentiality (127 comments); Noise & Distraction (125 comments); Food & Beverage (122 comments).
  After: Hardware (Monitors / Peripherals) (104 people, 23%); Noise & Distraction (97 people, 21%); Food & Beverage (97 people, 21%); Privacy & Confidentiality (94 people, 21%); Temperature / Thermal (77 people, 17%).
- **What's working.** Before: Reception & Entry (82% “positive”); Parking & Commute (74% “positive”); Community & Events (85% “positive”); Wellbeing / Wellness Spaces (77% “positive”); Support / Help Services (85% “positive”). After: Reception & Entry (55% positive, 49 people); Community & Events (53% positive, 29 people); Support / Help Services (62% positive, 23 people).
- **Red items by severity** (11; same set as before — status is unchanged): moderate: Focused individual work that requires concentration and minimal interruption, Availability of spaces that support focused work, Availability of spaces that support discretion, Thermal / temperature comfort · slight: Tasks requiring discretion, Availability of on-site food and beverage options (borderline), Virtual collaboration with remote or hybrid colleagues, Furniture comfort and ergonomics (borderline), Lighting comfort (borderline), Reliability of building operations, Cleanliness and upkeep of PulsePoint (borderline).

### VA-Richmond

- **Standing.** Before: rank 9 of 10, “Bottom third”. After: Index 67 ± 2, **Below the company average** (company 69.0); dimensions — Work Activity 62 ± 3 In line; Physical Feature 65 ± 2 In line; Service Level 68 ± 2 Below; Digital & Tech 71 ± 2 In line.
- **Top themes.** Before: Desk Availability & Reservations (141 comments); Food & Beverage (101 comments); Noise & Distraction (90 comments); Hardware (Monitors / Peripherals) (88 comments); Privacy & Confidentiality (80 comments).
  After: Food & Beverage (77 people, 26%); Noise & Distraction (65 people, 22%); Hardware (Monitors / Peripherals) (59 people, 20%); Privacy & Confidentiality (59 people, 20%); Desk Availability & Reservations (52 people, 18%).
- **What's working.** Before: Community & Events (80% “positive”). After: Entry and reception (rating, support 77); Communication about workplace updates, changes, or disruptions (rating, support 74).
- **Red items by severity** (10; same set as before — status is unchanged): moderate: Availability of spaces that support discretion, Availability of spaces that support focused work, Focused individual work that requires concentration and minimal interruption, Availability of on-site food and beverage options, Thermal / temperature comfort, Reliability of building operations · slight: Tasks requiring discretion, Virtual collaboration with remote or hybrid colleagues, Furniture comfort and ergonomics, Cleanliness and upkeep of PulsePoint.

## 5. Standing labels and multiple comparisons

Standing rule: Above / Below when the printed company value lies outside the site's 95% interval (2,000 respondent resamples within the site), otherwise In line. The labels below are what the cards print. The Holm check is for the record only and does not change any label.

| Site | Index | Work Activity | Physical Feature | Service Level | Digital & Tech |
|---|---|---|---|---|---|
| CA-Woodland Hills | Below | Below | Below | Below | In line |
| FL-Miami | In line | In line | In line | In line | Above |
| FL-Tampa | In line | In line | In line | In line | In line |
| GA-Atlanta | Above | Above | Above | Above | Above |
| IL-Chicago-Willis Tower | In line | In line | In line | Above | In line |
| IN-Indianapolis | In line | In line | Below | Above | Below |
| NY-New York (Penn1) | In line | In line | Above | In line | In line |
| OH-Mason | In line | In line | In line | In line | In line |
| VA-Norfolk | In line | In line | In line | In line | Above |
| VA-Richmond | Below | In line | In line | Below | In line |

Index and dimension labels that would **not** survive a Holm correction across all 50 site-by-measure comparisons (normal approximation from the bootstrap standard error): CA-Woodland Hills — Physical Feature (Below, Holm p = 0.275); FL-Miami — Digital & Tech (Above, Holm p = 0.506); IN-Indianapolis — Service Level (Above, Holm p = 0.513); NY-New York (Penn1) — Physical Feature (Above, Holm p = 0.462); VA-Norfolk — Digital & Tech (Above, Holm p = 0.960); VA-Richmond — Index (Below, Holm p = 0.245).

Associate-sentiment items (Q8 / Q12 / Q16 / Q19), site vs company:

| Site | Work Activity (Q8) | Physical (Q12) | Services (Q16) | Technology (Q19) |
|---|---|---|---|---|
| CA-Woodland Hills | Below | Below | Below | Below |
| FL-Miami | Above | In line | In line | In line |
| FL-Tampa | In line | In line | In line | In line |
| GA-Atlanta | Above | Above | Above | Above |
| IL-Chicago-Willis Tower | In line | In line | In line | In line |
| IN-Indianapolis | In line | In line | In line | In line |
| NY-New York (Penn1) | Above | In line | Above | In line |
| OH-Mason | Below | In line | In line | In line |
| VA-Norfolk | In line | In line | In line | In line |
| VA-Richmond | Below | In line | Below | In line |

Sentiment-item labels that would **not** survive a Holm correction across all 40 site-by-item comparisons (one-sample t-test against the company mean): CA-Woodland Hills — Physical Feature (Below, Holm p = 0.231); CA-Woodland Hills — Service Level (Below, Holm p = 0.066); CA-Woodland Hills — Digital & Tech (Below, Holm p = 0.922); FL-Miami — Work Activity (Above, Holm p = 0.922); NY-New York (Penn1) — Work Activity (Above, Holm p = 0.369); NY-New York (Penn1) — Service Level (Above, Holm p = 0.072); OH-Mason — Work Activity (Below, Holm p = 0.444); VA-Richmond — Work Activity (Below, Holm p = 1.000).

## 6. Comment themes — rules, precision, counts

Precision = share of a judged sample that is actually about the theme, read by the assistant (not yet confirmed by a person). “Before” = comments the old rule tagged; “after” = comments the new rule tags. Samples of 30 (fixed seed) unless the rule hits fewer, in which case every hit was judged. Rule-level rows judge only the comments the changed pattern hits.

| Rule | Before | After | Note |
|---|---:|---:|---|
| Desk Availability & Reservations | 9/30 (30%) | 26/30 (87%) | after v1 sample 21/30; tightened 'more desks' and amenity reservations, resampled (seed 11) |
| AV / Video Conferencing | 7/30 (23%) | 29/30 (97%) | after v1 24/30; dropped speaker(phone), restricted bare video, fixed 'room av'ailability; resampled seed 23 |
| Collaboration Space | 5/30 (17%) | 26/30 (87%) | after sample before sentence-boundary fix to the word window |
| In-Person / Team Presence | 21/30 (70%) | 25/30 (83%) | after v1 (bare in-person kept) 20/30; restricted 'in person' to working-together senses; resampled seed 11 |
| Space Availability & Crowding | 20/30 (67%) | 26/30 (87%) | then fixed 'space to'day and parking-lot 'is full' |
| Temperature / Thermal | 27/30 (90%) | 29/30 (97%) | then excluded 'freezing up' (devices) |
| Furniture & Ergonomics ('uncomfortable') | 14/30 (47%) | 15/15 (100%) | rule-level: only 15 comments match the restricted patterns |
| Building Operations ('broken', 'facility') | 13/30 (43%) | 29/30 (97%) | rule-level |
| RTO / Hybrid Policy ('hybrid', 'remote') | 22/30 (73%) | 27/30 (90%) | rule-level; on topic = discusses the writer's work arrangement or its requirements/effects. First after-sample (before restricting 'remote' to own arrangement) also 21/30 on the stricter reading |
| Positive / Appreciation ('great', 'love', 'best') | 12/30 (40%) | 26/30 (87%) | rule-level; on topic = actual praise/appreciation |
| Software / Tools ('system') | 20/30 (67%) | 24/30 (80%) | rule-level; then excluded immune / p.a. / mounting / numbering / number systems |
| Wellbeing ('nap') | 0/30 (0%) | no hits | old 'nap' hit 41 comments, mostly 'Indianapolis'; new rule (nap rooms/pods, place to nap) hits none |
| Cleanliness ('dust') | 17/19 (89%) | 16/16 (100%) | all hits judged; 'industry'/'industrial' removed |
| Navigation ('sign') | 6/30 (20%) | 4/4 (100%) | brief flagged 'sign in'; most remaining hits were etiquette signs, so 'sign' now needs a wayfinding context |
| Reception & Entry ('entry', 'check in') | 16/30 (53%) | 29/30 (97%) | 'check in' also matched 'checking a box' and desk-reservation check-ins; now reception check-ins only |
| Communication ('updates') | 4/26 (15%) | 2/2 (100%) | all hits judged; most were software/IT updates |
| Navigation ('lost') | 0/22 (0%) | no hits | all old hits were 'lost productivity/time'; new rule (get lost) has no hits |
| Connectivity ('connect') | 26/30 (87%) | 28/30 (93%) |  |

Theme counts on the 5,157 completed comments (all sites), old rules → new rules:

| Theme | Before | After | Change |
|---|---:|---:|---:|
| Privacy & Confidentiality | 884 | 884 | +0% |
| Noise & Distraction | 926 | 926 | +0% |
| Focus / Concentration | 557 | 557 | +0% |
| Meeting & Conference Rooms | 370 | 370 | +0% |
| Collaboration Space | 381 | 44 | -88% |
| In-Person / Team Presence | 163 | 155 | -5% |
| Desk Availability & Reservations | 1459 | 700 | -52% |
| Furniture & Ergonomics | 302 | 253 | -16% |
| Temperature / Thermal | 508 | 463 | -9% |
| Lighting | 203 | 203 | +0% |
| Space Availability & Crowding | 403 | 296 | -27% |
| Wellbeing / Wellness Spaces | 140 | 102 | -27% |
| Food & Beverage | 702 | 702 | +0% |
| Cost / Pricing | 206 | 206 | +0% |
| Cleanliness & Upkeep | 371 | 369 | -1% |
| Reception & Entry | 235 | 220 | -6% |
| Community & Events | 181 | 181 | +0% |
| Connectivity (WiFi / Login / Power) | 561 | 535 | -5% |
| Hardware (Monitors / Peripherals) | 938 | 938 | +0% |
| AV / Video Conferencing | 353 | 136 | -61% |
| Software / Tools | 196 | 185 | -6% |
| Building Operations | 249 | 144 | -42% |
| Parking & Commute | 333 | 333 | +0% |
| Navigation / Wayfinding | 149 | 100 | -33% |
| Communication & Updates | 111 | 89 | -20% |
| Support / Help Services | 128 | 128 | +0% |
| RTO / Hybrid Policy | 486 | 330 | -32% |
| Positive / Appreciation | 714 | 609 | -15% |
| Overall Dissatisfaction | 169 | 169 | +0% |

Unclassified share: 9.4% → 12.3%. No catch-all patterns were added to hide it. Desk Availability fell 52% (expected 40–50%), AV 61% (expected more than half). Five gold-set rows are no-content answers (“None that I can think of”, thanks only) that the junk filter does not catch — not changed in this pass.

## 7. Sentiment

- Prior fix (2.4): 409 of 5,259 non-junk labels changed (7.8%); card population 379 of 4,939 (7.7%). Physical Limitations: 259 Neutral/Mixed → Negative, 74 Positive → Neutral/Mixed; Work Types Not Supported: 38 and 38. Card-population mix Negative 46.3% → 51.9%, Neutral/Mixed 32.1% → 28.6%, Positive 21.6% → 19.5%.
- Accuracy against the gold set (`data/gold/eval_report.md`): **verified labels: none yet** — accuracy is not established. Draft labels (300, pre-filled by the assistant, labeller `agent-draft`): sentiment 63% (shipped labels at 45dba0b: 60%); 109 of 111 errors are more positive than the comment. Themes: micro F1 68% (precision 69%, recall 67%); the pre-4A rules scored precision 59%, F1 63% on the same rows.
- The local-model tagger is built but was tested only against a stub (no local model reachable here); it is off.

## 8. Stored action plans

Not regenerated; priorities and their order unchanged. Every edited line, old → new: [`docs/plan-evidence-refresh-2026-11.md`](plan-evidence-refresh-2026-11.md). Priorities and protect items the corrected evidence no longer clearly supports (left in place, shown on each card):

- **CA-Woodland Hills** — Priority 5 (Meeting-room and video-conferencing reliability): the AV / Video Conferencing comments behind it fall from 39 to 14 (14 people) once bare mentions of Microsoft Teams stop counting; the priority now rests on the Under Supported virtual-collaboration item and Meeting & Conference Rooms comments.
- **CA-Woodland Hills** — Protect item 'Community & Events': the true positive share is 29% (19% negative) — it no longer clears the What's-working bar
- **CA-Woodland Hills** — Protect item 'Cleanliness & Upkeep': the true positive share is 47% (28% negative) — it no longer clears the What's-working bar
- **FL-Miami** — Priority 3 (Network connectivity reliability): connectivity comments are 35% negative (17 comments from 12 people) and the protect list also calls connectivity a strength for many; the case for a site-wide reliability problem is thin — confirm the specific floors or spaces before acting.
- **FL-Miami** — Protect item 'Desk Availability & Reservations': the true positive share is 33% (33% negative) — it no longer clears the What's-working bar
- **FL-Tampa** — Protect item 'Reception & Entry': the true positive share is 30% (30% negative) — it no longer clears the What's-working bar
- **GA-Atlanta** — Protect item 'Collaboration Space': the true positive share is 20% (20% negative) — it no longer clears the What's-working bar
- **GA-Atlanta** — Protect item 'AV / Video Conferencing': the true positive share is 9% (55% negative) — it no longer clears the What's-working bar
- **GA-Atlanta** — Protect item 'Parking & Commute': the true positive share is 30% (33% negative) — it no longer clears the What's-working bar
- **IN-Indianapolis** — Priority 2 (Desk reservations and booking window): the two-week booking window it centres on is mentioned in 11 comments from 9 people; the figure it cited (375 comments) counted every mention of a desk. After the re-tag Desk Availability & Reservations is still raised by 147 people (21% of respondents), so desk booking remains material, but the booking-window change itself rests on a small number of comments — validate locally first.
- **IN-Indianapolis** — Protect item 'Community & Events': the true positive share is 45% (20% negative) — it no longer clears the What's-working bar
- **NY-New York (Penn1)** — Priority 5 (Meeting-room hookups, guides and booking): the AV / Video Conferencing leg is 5 comments from 4 people; Meeting & Conference Rooms (12 comments, 12 people) and the Under Supported virtual-collaboration item carry it.
- **OH-Mason** — Priority 5 (Meeting-room booking discipline and in-room connections): AV / Video Conferencing comments fall from 35 to 10 (10 people) after the re-tag; the in-room connections half of this priority is thinly supported by comments — the booking-discipline half (Meeting & Conference Rooms, 44 people) is not affected.
- **OH-Mason** — Protect item 'Community & Events': the true positive share is 33% (17% negative) — it no longer clears the What's-working bar
- **VA-Norfolk** — Priority 5 (Restroom repairs and building upkeep): Building Operations comments fall from 51 to 32 (29 people) and are 31% negative; the priority now rests mainly on the two Under Supported survey items (building operations and cleanliness), both with low Friction (13 and 12).
- **VA-Norfolk** — Protect item 'Parking & Commute': the true positive share is 41% (28% negative) — it no longer clears the What's-working bar
- **VA-Norfolk** — Protect item 'Wellbeing / Wellness Spaces': the true positive share is 41% (27% negative) — it no longer clears the What's-working bar
- **VA-Richmond** — Protect item 'Community & Events': the true positive share is 44% (24% negative) — it no longer clears the What's-working bar

## 9. Open decisions for the owner

- **Analyst exports keep role and tenure.** The offline dashboard, CSV and Excel exports still show role and tenure beside comments (now marked “internal analysis only”). Decide whether to apply the attribution rule there too.
- **Prompt v3.2 sign-off.** v3.2 is recorded in `_meta` as a draft; it needs work-stream sign-off before anyone regenerates plans.
- **Model classifier.** `llm_tagger.py` is off. It has not been evaluated on real model output; switch it on only if it beats the rules on verified gold labels.
- **Gold labels need a person.** All 300 rows are `agent-draft`; until at least 30 are verified, cards say tagging is not yet validated.
- **The original raw file is still in git history** (commits up to `45dba0b`). This pass did not rewrite history.
- **Names inside comment text** are not redacted in this pass (quotes, the comment list, stored plans, `comments.csv`, `emerging_themes.json`).
- **Hosted model calls** were not exercised (no keys here). New defaults follow vendor docs, but newer reasoning models may reject `temperature: 0`; test once before relying on a hosted provider.
- **`emerging_themes.json`** was regenerated with this environment's scikit-learn after the re-tag; cluster sizes differ from a rebuild on another machine.
