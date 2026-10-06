from __future__ import annotations

import argparse
import random
from pathlib import Path

SUBJECTS = [
    "המערכת", "המשתמש", "הצוות", "המחשב", "היישום", "הפרויקט", "המסמך", "הטבלה",
    "השרת", "הקובץ", "החוקר", "המפתח", "המנהל", "התהליך", "המודל", "הנתון",
]
VERBS = [
    "שומר", "קורא", "מעדכן", "בודק", "מסכם", "מציג", "מחשב", "משווה", "ממיין",
    "מתכנן", "מפעיל", "מנתח", "מארגן", "מעתיק", "מתעד", "מסנן",
]
OBJECTS = [
    "את הנתונים", "את הרשומה", "את התוצאה", "את הקובץ", "את המשימה", "את הרשימה",
    "את ההגדרות", "את הדוח", "את הטבלה", "את המידע", "את הבקשה", "את השלבים",
]
ADVERBS = [
    "בזהירות", "באופן אוטומטי", "במהירות", "לפי הכללים", "לפי הסדר", "באופן עקבי",
    "לאחר בדיקה", "לפני השמירה", "במהלך העבודה", "לפי הצורך",
]
CONNECTORS = ["לאחר מכן", "בנוסף", "בשלב הבא", "במקביל", "לבסוף", "כאשר נדרש"]

FACTS = [
    "מים קופאים בטמפרטורה נמוכה ומתאדים כאשר הם מתחממים.",
    "עץ גדל לאורך זמן וזקוק לאור, מים וחומרי הזנה.",
    "מחשב מבצע הוראות לפי תוכנה ויכול לשמור מידע בקבצים.",
    "לוח שנה מחלק את הזמן לימים, שבועות וחודשים.",
    "מפה מציגה מקומות ודרכים ומסייעת בתכנון מסלול.",
    "טבלה מסדרת מידע בשורות ובעמודות כדי להקל על השוואה.",
    "בדיקה חוזרת יכולה לזהות שגיאות לפני פרסום תוצאה.",
    "גיבוי שומר עותק נוסף של מידע חשוב למקרה של תקלה.",
    "סיסמה חזקה עדיפה כאשר היא ארוכה וקשה לניחוש.",
    "מדידה עקבית מאפשרת להשוות תוצאות בין זמנים שונים.",
]


def sentence(rng: random.Random) -> str:
    if rng.random() < 0.25:
        return rng.choice(FACTS)
    s = f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)} {rng.choice(ADVERBS)}."
    if rng.random() < 0.45:
        s += f" {rng.choice(CONNECTORS)}, {rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)}."
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=int, default=30000)
    ap.add_argument("--out", default="hebrew_bootstrap_corpus.txt")
    args = ap.parse_args()
    rng = random.Random(20261006)
    path = Path(args.out)
    with path.open("w", encoding="utf-8") as f:
        for _ in range(args.lines):
            f.write(sentence(rng) + "\n")
    print(f"wrote={path} lines={args.lines} bytes={path.stat().st_size}")


if __name__ == "__main__":
    main()
