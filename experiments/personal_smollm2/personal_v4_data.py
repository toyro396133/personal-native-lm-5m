"""Fictional chat-history benchmark data. Never use actual user information."""
FACTS = [
 # source utterance, train question, unseen test question, answer, plausible alternative
 ("Could you call my side project Copper Finch from now on?", "What do I call my side project?", "Remind me, what's the nickname for that side project?", "Copper Finch", "Silver Otter"),
 ("I work best in focus blocks of 35 minutes.", "How many minutes is my focus block?", "What duration do I use for a focused work block?", "35 minutes", "45 minutes"),
 ("For progress updates, I like three bullet points rather than a long report.", "What layout do I like for progress updates?", "How should you format a project progress update for me?", "three bullet points", "a long report"),
 ("My notes archive has a name: Lantern Vault.", "What is my notes archive named?", "What did I call the archive where I keep notes?", "Lantern Vault", "Atlas Shelf"),

 ("I want the editor's appearance set to dark mode.", "What appearance do I want in the editor?", "Which editor theme did I settle on?", "dark mode", "light mode"),
 ("The dashboard I've been drawing has amber accents.", "What accent color is on my dashboard?", "Which color accent did I choose for the dashboard?", "amber", "violet"),
 ("The separate meeting notes folder is called Atlas Shelf.", "What name did I give the meeting notes folder?", "Where did I say my meeting notes are filed?", "Atlas Shelf", "Lantern Vault"),
 ("Sunday is when I review my current tasks each week.", "When do I review my tasks?", "Which weekday is my recurring task review?", "Sunday", "Friday"),

 ("I put on rain sounds when I'm doing focused writing.", "What sound do I play for focused writing?", "What background audio do I prefer when writing?", "rain sounds", "soft piano"),
 ("Please schedule my weekly reminder for Tuesday morning.", "When do I prefer the weekly reminder?", "What daypart should that recurring reminder use?", "Tuesday morning", "Thursday evening"),
 ("I'd like project sprints to last two weeks.", "How long are the project sprints I prefer?", "What's my usual sprint duration?", "two weeks", "one week"),
 ("When you outline a roadmap, I'd rather see a timeline than a mind map.", "What format do I prefer for roadmaps?", "How should my roadmap be displayed?", "a timeline", "a mind map"),

 ("I refer to my backlog board as Harbor Board.", "What is my backlog board called?", "What name did I give the backlog?", "Harbor Board", "Copper Finch"),
 ("For my data exports, use CSV files by default.", "What is my default export format?", "Which file format do I usually request for exporting data?", "CSV", "PDF"),
 ("For drawing system diagrams, I usually work in draw.io.", "What diagram tool do I use?", "Which application do I draw diagrams in?", "draw.io", "Figma"),
 ("Show me worked examples first when I learn a new concept.", "How do I like to learn new concepts?", "What's my preferred starting point for a new topic?", "worked examples", "abstract theory"),

 ("The terminal I use for local commands is PowerShell.", "What terminal do I use?", "Which command shell do I normally work in?", "PowerShell", "Bash"),
 ("The test runner for my toy Python app is pytest.", "What tool runs my Python tests?", "Which test framework did I pick for the demo app?", "pytest", "unittest"),
 ("Label the issues I will tackle next with next-action.", "What label identifies my next tasks?", "How do I tag issues that should be worked on next?", "next-action", "later"),
 ("In bug reports I want reproduction steps before the screenshots.", "What should go first in my bug reports?", "What information do I want before screenshots in a bug report?", "reproduction steps", "a summary"),

 ("If I send an article to read, give me short summaries.", "What kind of reading summaries do I prefer?", "How detailed should summaries of articles be for me?", "short summaries", "full transcripts"),
 ("My study session starts at 8 AM.", "At what time does my study session begin?", "What hour did I choose for my study block?", "8 AM", "10 AM"),
 ("I make the daily plan in a paper notebook, not an app.", "Where do I write my daily plan?", "What do I use to write out today's plan?", "a paper notebook", "a mobile app"),
 ("For the demo travel dataset, the reference city is Oslo.", "What city is used in my demo dataset?", "Which city did I pick as the example in my travel demo?", "Oslo", "Helsinki"),
]

# Later messages contradict old values. Evaluations must use the latest truth.
UPDATES = [
    {"stage": 4, "index": 1, "message": "Update on my focus routine: I've changed my blocks from 35 minutes to 25 minutes.", "answer": "25 minutes", "wrong": "35 minutes"},
    {"stage": 5, "index": 5, "message": "One correction about my dashboard: its accent is teal now, not amber.", "answer": "teal", "wrong": "amber"},
]

# Irrelevant user requests; several share misleading words with relevant facts.
DISTRACTORS = [
  "Can you describe how attention layers work in a transformer?",
  "What's the difference between a sprint review and sprint planning?",
  "I read a piece about dashboards in public transport systems.",
  "Would you explain the difference between CSV and JSON?",
  "Please tell me why people use dark themes for code editors.",
  "I'm thinking about using a notebook to sketch a database schema.",
  "How does retrieval augmented generation differ from fine tuning?",
  "Can you give examples of good bug report structures?",
  "I saw a video about hiking near Oslo and Bergen.",
  "What is the difference between two-week and three-week release cycles?",
  "What makes a useful progress report for a large engineering team?",
  "Tell me about the history of project management software.",
]

GENERIC_QA = [
    ("What is the capital of France?", "Paris"),
    ("What is five plus three?", "8"),
    ("How many days are in a week?", "7"),
    ("What gas do people breathe to survive?", "oxygen"),
    ("What is the opposite of hot?", "cold"),
    ("What is the chemical formula for water?", "H2O"),
    ("Which planet is known as the red planet?", "Mars"),
    ("What is the largest ocean on Earth?", "Pacific"),
]
