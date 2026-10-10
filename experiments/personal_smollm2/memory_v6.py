"""Auditable synthetic preference memory. Bounded language grammar; not generic NLP.

Current state is separated from provenance. Temporary overrides are consumed by
eligible response queries; deletion removes active state but preserves audit
metadata in this research fixture (real privacy deletion needs stronger policies).
"""
import re
from dataclasses import dataclass, field

STYLES = ("two_bullets", "three_steps", "one_sentence")
STYLE_TEXT = {
    "two_bullets": "two brief bullet points",
    "three_steps": "three numbered steps",
    "one_sentence": "one concise sentence",
}
ALIASES = {
    "two brief bullet points": "two_bullets",
    "two bullets": "two_bullets",
    "a pair of bullet points": "two_bullets",
    "three numbered steps": "three_steps",
    "three steps": "three_steps",
    "one concise sentence": "one_sentence",
    "a single sentence": "one_sentence",
    "one sentence": "one_sentence",
}
ASSERT = re.compile(
    r"^(?:please )?(?:answer my practical questions|reply to my how-to questions|format your answers for me)"
    r" (?:in|as|with) (.+?)\.?$", re.I)
PREFER = re.compile(
    r"^i prefer (.+?) (?:in|for) (?:my practical answers|practical advice replies)\.?$", re.I)
CORRECT = re.compile(
    r"^(?:actually,? |i changed my mind: )"
    r"(?:use|please use) (.+?) (?:from now on|for my practical answers)\.?$", re.I)
TEMP = re.compile(
    r"^for (?:my|the) next (\d+) (?:replies|answers),? (?:use|reply in) (.+?)\.?$", re.I)
DELETE = re.compile(
    r"^(?:forget|delete|remove) my (?:response format|reply style) preference\.?$", re.I)
HYPOTHETICAL = re.compile(r"\b(might|maybe|someday|considering|would consider|perhaps|if i)\b", re.I)
THIRD_PARTY = re.compile(r"^(?:my colleague|my teammate|my friend|the blog|the article|someone|my teacher)\b",re.I)
QUOTE = re.compile(r"""["'“”]""")
LEAKY = re.compile(r"\b(my name is|password|secret|token|api key)\b",re.I)

@dataclass
class Memory:
    active: str | None = None
    previous: list = field(default_factory=list)
    pending: list = field(default_factory=list)
    temporary: str | None = None
    remaining: int = 0
    event_index: int = 0
    audit: list = field(default_factory=list)

    def record(self,text,kind,value=None):
        self.audit.append({"event":self.event_index,"kind":kind,"value":value,
                           "source":text})
        return kind

    def observe(self,text,role="user"):
        self.event_index+=1
        text=text.strip()
        if role!="user":
            return self.record(text,"ignored_assistant")
        if LEAKY.search(text):
            return self.record("[REDACTED]","rejected_sensitive")
        if THIRD_PARTY.search(text) or QUOTE.search(text):
            return self.record(text,"ignored_quoted_or_third_party")
        if HYPOTHETICAL.search(text):
            return self.record(text,"ignored_hypothetical")
        if DELETE.fullmatch(text):
            if self.active is not None:
                self.previous.append({"value":self.active,"superseded_at":self.event_index,
                                      "reason":"deleted"})
            self.active=None
            self.temporary=None
            self.remaining=0
            return self.record(text,"deleted")
        match=TEMP.fullmatch(text)
        if match:
            length=int(match.group(1))
            style=ALIASES.get(match.group(2).lower().rstrip("."))
            if style is None or not 1<=length<=5:
                return self.record(text,"needs_clarification")
            self.temporary=style
            self.remaining=length
            return self.record(text,"temporary",style)
        match=ASSERT.fullmatch(text) or PREFER.fullmatch(text) or CORRECT.fullmatch(text)
        if match:
            style=ALIASES.get(match.group(1).lower().rstrip("."))
            if style is None:
                return self.record(text,"needs_clarification")
            if self.active is not None and self.active!=style:
                self.previous.append({"value":self.active,"superseded_at":self.event_index,
                                      "reason":"updated"})
            self.active=style
            return self.record(text,"set",style)
        if re.search(r"\b(?:prefer|my style|my preference|format your answers)\b",text,re.I):
            self.pending.append({"event":self.event_index,"source":text})
            return self.record(text,"needs_clarification")
        return self.record(text,"ignored_non_preference")

    def peek(self):
        return self.temporary if self.remaining>0 else self.active

    def consume(self):
        current=self.peek()
        if self.remaining>0:
            self.remaining-=1
            if self.remaining==0:
                self.temporary=None
        return current

def advice_task(question):
    """Conservative regex gate: matches explicit advice request but no generic trivia."""
    return bool(re.search(r"^(?:how (?:can|should|do) i |what steps should i |help me |give me advice on )",
                          question.lower()))

def prompt_for_style(style):
    if style is None:
        return ""
    return f"Please respond in {STYLE_TEXT[style]}. "

def memory_smoke():
    m=Memory()
    events=[
      ("I prefer two bullets for practical advice replies.","set","two_bullets"),
      ("My teammate prefers one sentence.","ignored_quoted_or_third_party","two_bullets"),
      ("I might prefer three numbered steps someday.","ignored_hypothetical","two_bullets"),
      ('An article stated "I prefer one sentence".',"ignored_quoted_or_third_party","two_bullets"),
      ("Actually, use three numbered steps from now on.","set","three_steps"),
      ("Forget my response format preference.","deleted",None),
      ("For my next 2 replies, use one concise sentence.","temporary","one_sentence"),
    ]
    for text,kind,current in events:
        assert m.observe(text)==kind,(text,m.audit[-1])
        assert m.peek()==current,(text,m.peek(),current)
    assert m.consume()=="one_sentence"
    assert m.consume()=="one_sentence"
    assert m.peek() is None
    assert m.observe("Please answer my practical questions in two bullets.")=="set"
    assert m.peek()=="two_bullets"
    assert m.observe("I prefer an elaborate diagram.")=="needs_clarification"
    assert m.peek()=="two_bullets"
    assert m.observe("my password is 1234")=="rejected_sensitive"
    assert "[REDACTED]" in m.audit[-1]["source"]
    assert advice_task("How can I organize my desk?")
    assert not advice_task("What is the capital of France?")
    return {"events":len(m.audit),"revisions":len(m.previous),
            "unresolved":len(m.pending)}

if __name__=="__main__":
    print(memory_smoke())
