"""Keyword-rule regressions (Nov 2026 review). Every sentence here is invented."""
import pytest

from taxonomy import tag_themes

DESK = "Desk Availability & Reservations"
AV = "AV / Video Conferencing"
COLLAB = "Collaboration Space"
PRESENCE = "In-Person / Team Presence"
CROWD = "Space Availability & Crowding"
TEMP = "Temperature / Thermal"
FURN = "Furniture & Ergonomics"
BLDG = "Building Operations"
RTO = "RTO / Hybrid Policy"
POS = "Positive / Appreciation"
SOFT = "Software / Tools"
CLEAN = "Cleanliness & Upkeep"
WELL = "Wellbeing / Wellness Spaces"
NAV = "Navigation / Wayfinding"
RECEP = "Reception & Entry"
COMMS = "Communication & Updates"
CONN = "Connectivity (WiFi / Login / Power)"


def themes(text):
    return set(tag_themes(text)[0])


# (sentence, theme that must NOT be tagged)
FALSE_POSITIVES = [
    # desk: a mention of a desk is not an availability / booking issue
    ("The help desk fixed my laptop in ten minutes.", DESK),
    ("The front desk staff always say good morning.", DESK),
    ("The security desk is staffed until six.", DESK),
    ("I sit at my desk on calls most of the day.", DESK),
    ("My cube is next to the kitchen and it smells like popcorn.", DESK),
    ("We would like more quiet seating near the windows.", DESK),
    # AV: Teams the app, or "our teams"
    ("I spend all day on Teams calls from a phone booth.", AV),
    ("Our teams sit on three different floors.", AV),
    ("The support teams resolve tickets quickly.", AV),
    ("People take calls on speakerphone in the open area.", AV),
    ("The room availability page is hard to read.", AV),
    # collaboration: presence, not space
    ("Nobody I collaborate with is ever in this office.", COLLAB),
    ("I collaborate with people in other states, so I just sit alone.", COLLAB),
    # crowding: full-time, full day, full-service, parking lot
    ("I work full-time and come in a full day each week.", CROWD),
    ("The full-service cafe used to be open on Fridays.", CROWD),
    ("I usually book a space today before nine.", CROWD),
    ("The parking lot is full by nine.", CROWD),
    # temperature: food, drinks, desks, welcomes
    ("I miss having hot food at lunch.", TEMP),
    ("The hot chocolate machine is empty.", TEMP),
    ("Please add a hot water tap at the coffee bar.", TEMP),
    ("Hot desking makes it hard to find my team.", TEMP),
    ("The cold water dispenser leaks.", TEMP),
    ("The staff give everyone a warm welcome.", TEMP),
    ("There is nowhere to heat up lunch on my floor.", TEMP),
    ("My monitor keeps freezing up in the afternoon.", TEMP),
    # furniture: uncomfortable about something else
    ("I feel uncomfortable taking private calls in an open cube.", FURN),
    ("It is uncomfortable when someone sits in my reservation.", FURN),
    # building operations: broken hardware, "facility" = the building
    ("My keyboard is broken and the monitor flickers.", BLDG),
    ("The facility is beautiful and bright.", BLDG),
    # RTO: room names and remote/hybrid colleagues
    ("The hybrid huddle rooms on four are great for small groups.", RTO),
    ("My remote colleagues join every meeting by video.", RTO),
    ("Most of my hybrid colleagues come in on Tuesdays.", RTO),
    ("All the others on my team are remote, so I sit alone.", RTO),
    # positive: wishes and negations
    ("It would be great to have more snacks.", POS),
    ("The coffee is not great.", POS),
    ("I would love more quiet rooms.", POS),
    # software: HVAC / AC / parking systems
    ("The HVAC system is loud on the third floor.", SOFT),
    ("The parking system charges me twice.", SOFT),
    ("The AC system shuts off at five.", SOFT),
    ("My immune system is weak, so I wipe the desk down.", SOFT),
    # substring bugs
    ("I have worked in this industry for twenty years.", CLEAN),
    ("The napkins ran out and I had to snap at nobody.", WELL),
    ("The Indianapolis office has a nice cafe.", WELL),
    ("It takes a long time to sign in at a new desk.", NAV),
    ("Commuting means lost productivity every week.", NAV),
    ("I do data entry all day.", RECEP),
    ("I check in with my team every morning.", RECEP),
    ("Coming in feels like checking a box.", RECEP),
    ("Software updates restart my laptop during meetings.", COMMS),
    ("I booked a connect capsule for my one-on-one.", CONN),
    ("I want to connect with people on other teams.", CONN),
]

# (sentence, theme that MUST be tagged)
TRUE_POSITIVES = [
    ("I could not find a desk on Tuesday.", DESK),
    ("Someone was sitting in my reserved desk again.", DESK),
    ("The desk reservation window is only two weeks.", DESK),
    ("Hoteling means I never know where I will sit.", DESK),
    ("There are not enough desks on busy days.", DESK),
    ("The conference room camera never works.", AV),
    ("Screen sharing in the huddle rooms is unreliable.", AV),
    ("We need more collaboration spaces with whiteboards.", COLLAB),
    ("Nobody I collaborate with is ever in this office.", PRESENCE),
    ("The phone booths are always full.", CROWD),
    ("The office is too cold every afternoon.", TEMP),
    ("The chairs are uncomfortable after an hour.", FURN),
    ("The escalator is broken again.", BLDG),
    ("I would prefer to work remotely full time.", RTO),
    ("The hybrid schedule does not fit my role.", RTO),
    ("I love the new lobby.", POS),
    ("The reservation system is slow.", SOFT),
    ("The floors are dusty.", CLEAN),
    ("A nap room would help on long days.", WELL),
    ("There are no signs to direct visitors.", NAV),
    ("The entry doors are too heavy.", RECEP),
    ("Please keep us updated on the renovation.", COMMS),
    ("I cannot connect to the WiFi on the fifth floor.", CONN),
]


@pytest.mark.parametrize("text,theme", FALSE_POSITIVES)
def test_false_positive_families(text, theme):
    assert theme not in themes(text)


@pytest.mark.parametrize("text,theme", TRUE_POSITIVES)
def test_true_positives_still_tag(text, theme):
    assert theme in themes(text)
