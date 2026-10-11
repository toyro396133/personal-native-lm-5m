"""Three independent fictional user profiles. Explicit preference syntax is deliberate:
this test isolates versioning, routing, and training; it does not test free-text extraction.
"""
SLOTS = [
    "focus duration","dashboard accent","report layout","note archive",
    "project alias","weekly review day","editor theme","export format",
    "study start time","planning tool","diagram tool","background sound",
]
VALUES = [
    ["35 minutes","amber","three bullet points","Lantern Vault","Copper Finch","Sunday","dark mode","CSV","8 AM","paper notebook","draw.io","rain sounds"],
    ["40 minutes","violet","one paragraph","North Shelf","Silver Heron","Tuesday","light mode","JSON","7 AM","kanban board","Figma","soft piano"],
    ["20 minutes","olive","a short table","Atlas Drawer","Blue Orbit","Friday","high contrast","TSV","9 AM","whiteboard","Excalidraw","ocean waves"],
]
UPDATES = [
    [(0,"25 minutes"),(1,"teal")],
    [(0,"30 minutes"),(1,"coral")],
    [(0,"45 minutes"),(1,"navy")],
]
DISTRACTORS = [
    "Can you explain how transformers handle long context?",
    "I read an article about modern dashboard design patterns.",
    "What is the difference between CSV, TSV and JSON files?",
    "A tutorial discussed the history of project management software.",
    "Could you explain the physics of ocean waves?",
    "I am curious about why time management can improve focus.",
]
GENERIC_QA = [
    ("What is the capital of France?","Paris"),
    ("What is five plus three?","8"),
    ("How many days are in a week?","7"),
    ("What gas do humans breathe?","oxygen"),
    ("What is the chemical formula for water?","H2O"),
    ("Which planet is called the red planet?","Mars"),
    ("What is the opposite of hot?","cold"),
    ("What is the capital of Italy?","Rome"),
    ("In my maths homework, what is six plus two?","8"),
    ("In my science book, what is the formula for water?","H2O"),
]

def utterances(profile,stage):
    """User messages have no answers/test labels, but use constrained authored grammar."""
    rows=[]
    for i,slot in enumerate(SLOTS[:4*min(stage,3)]):
        rows.append({"stage":(i//4)+1, "role":"user",
                     "text":f"For {slot}, my choice is {VALUES[profile][i]}."})
    for t in range(1,stage+1):
        for j in range(2):
            rows.append({"stage":t,"role":"user","text":DISTRACTORS[(2*t+j+profile)%len(DISTRACTORS)]})
    if stage>=3:
        for index,new_value in UPDATES[profile]:
            rows.append({"stage":3,"role":"user",
                         "text":f"Correction for {SLOTS[index]}: use {new_value} instead of {VALUES[profile][index]}."})
    return sorted(rows,key=lambda r:r["stage"])

def test_rows(profile,stage):
    rows=[]
    for i,slot in enumerate(SLOTS[:4*min(stage,3)]):
        value=VALUES[profile][i]
        if stage>=3:
            for k,v in UPDATES[profile]:
                if i==k:value=v
        rows.append({"id":f"s{i}","slot":slot,"right":value,
                     "test":f"What is my {slot}?",
                     "train":f"Remind me of my chosen {slot}.",
                     "wrong":VALUES[(profile+1)%3][i],
                     "old":VALUES[profile][i] if value!=VALUES[profile][i] else None})
    return rows
