# COMPASS V9.15 — תיוג אנושי של החלטות אמיתיות מחכם לב

**מטרת המחקר:** ללמד את COMPASS לזהות מתוך תיעוד עבודה את ההבדל בין עיקרון מאושר, הצעה לא מאושרת, תיאור היסטורי, תיקון או מצב לא ברור — בלי לקבל מראש תוויות מושלמות.

**מצב נוכחי:** נבנה מחלץ מבוסס מקורות, נבנה מסווג נלמד קטן, ונבנה כלי תיוג אנושי מקומי. בניסוי הראשון על מסמכי חכם לב התקבלה בעיית איכות נתונים: בחומר האימון המסומן במפורש בכותרות היו 5 דוגמאות מאושרות, 2 היסטוריות, ו־0 דוגמאות הצעה שניתן לאמן עליהן. לפיכך המסווג **מסרב להתחיל אימון** ואין תוצאת למידה תקפה. זו אינה הצלחה סינתטית ואינה פריצת דרך.

## יצירת מסך תיוג פרטי במחשב (Windows PowerShell)

התקן Python 3.11+ (ללא חבילות נוספות). ודא שמאגרי COMPASS וחכם לב משוכפלים מקומית. עבור לתיקיית מאגר COMPASS:

```powershell
git fetch origin
git switch experiment/compass-v9-15
```

מתוך התיקייה שבה נמצא הקוד, צור דף תיוג פרטי. **הקובץ מכיל ציטוטים מתוך הפרויקט הפרטי ולכן יש לשמור אותו על המחשב בלבד, לא בגיטהאב הציבורי**:

```powershell
python .\experiments\compass_v9_15\human_review.py `
  --repo "$env:USERPROFILE\Desktop\chacham-lev" `
  --ref fe9863278a04dedf27a0c4fcb8cca56b91ec4544 `
  --project chacham-lev `
  --path docs/product/importance-map.md `
  --path docs/campaigns/telephone-call-outcomes-and-statuses-proposal-2026-10-11.md `
  --path docs/campaigns/telephone-core-closure-review-2026-10-11.md `
  --path docs/campaigns/operational-interview-live-notes.md `
  --output "$env:USERPROFILE\Desktop\compass-v915-private-review.html" `
  --limit 60
```

כדי לפתוח:

```powershell
Start-Process "$env:USERPROFILE\Desktop\compass-v915-private-review.html"
```

בממשק, הבודק קורא את הכותרת ואת העדות שמתחתיה, מסמן מעמד החלטה, מסביר את הנימוק ומציין מתי דרוש בירור. לאחר מכן מורידים קובץ תיוגים מקומי. אף סיווג אינו משנה עיקרון מאושר ב־COMPASS. הכלי אינו שולח שום נתון לרשת.

**חשוב למחקר:** אחרי תיוג עצמאי וסקירת הסכמות בין בודקים, יש לנעול קבוצת בדיקה נפרדת שלא תשמש באימון. רק אז נוכל לבדוק למידה מול בקרות מילוליות על החלטות חדשות.

## מחלץ המקורות של V9.14

כדי ליצור רשימת הצעות עם מקור שורה מדויק מתוך מסמכים פרטיים:

```powershell
python .\experiments\compass_v9_14\decision_miner.py `
  --repo "$env:USERPROFILE\Desktop\chacham-lev" `
  --ref fe9863278a04dedf27a0c4fcb8cca56b91ec4544 `
  --project chacham-lev `
  --path docs/campaigns/telephone-call-outcomes-and-statuses-proposal-2026-10-11.md `
  --private-output "$env:USERPROFILE\Desktop\compass-v914-private-candidates.json" `
  --sanitized-metrics "$env:USERPROFILE\Desktop\compass-v914-metrics.json"
```

הקובץ המלא מכיל תוכן פרטי וחייב להישאר מקומי. כל רשומה היא `PENDING_REVIEW`, לא החלטה מאושרת; מצב `APPROVED_CLAIM` פירושו רק שבכותרת *נטען* כי התקבל אישור.

## שלב נוסף — העברת הצעות לזיכרון COMPASS בלי לאשר אותן

לאחר יצירת קובץ `compass-v914-private-candidates.json` לפי הפקודה למעלה, בדוק קודם במצב **קריאה בלבד** אילו הצעות אפשר להכניס:

```powershell
python .\experiments\compass_v9_15\bridge_pending.py `
  --private-candidates "$env:USERPROFILE\Desktop\compass-v914-private-candidates.json" `
  --db "$env:USERPROFILE\Desktop\compass-chacham.sqlite3" `
  --project chacham-lev
```

כברירת מחדל הקובץ אינו משנה דבר. כדי ליצור **הצעות בלבד** במסד, הוסף את הדגל `--import-pending` לאותה פקודה. אף הצעה לא מאושרת אוטומטית, אפילו אם בכותרת המקור כתוב ״החלטה מאושרת״.

לאחר מכן אפשר לפתוח את ממשק הסקירה המקומי:

```powershell
python .\experiments\compass_v9_12\compass_ui.py `
  --db "$env:USERPROFILE\Desktop\compass-chacham.sqlite3" `
  --port 8787 --open
```

לפני אישור, יש לעיין במקור ולאמת מי הכריע, מהי ההחלטה המדויקת, מהו הנימוק, האם היא עוד בתוקף והאם היא מתקנת החלטה קודמת. **קובצי המקור, התיוגים ומסד הנתונים נשארים פרטיים במחשב.**

## מגבלות

- מרבית המסמכים האמיתיים אינם מתויגים במפורש בכל פסקה; מערכת זהירה צריכה לדעת להימנע.
- מסווג הכותרות אינו מודל שפה והוא אינו מאמת מי באמת אישר.
- מסווג התוכן V9.15 משתמש בתוויות כסף מתוך כותרות; זה אינו בוחן אנושי עצמאי.
- עדיין לא אומן מודל COMPASS מוצלח מתוך שיחות חופשיות של פרויקט אמיתי.
- זהו מחקר **COMPASS** בלבד; אינו משנה את מודל SELF או את מערכת חכם לב.

[פרוטוקול המחקר](PROTOCOL_V915.md) · [רכיב הלמידה](weak_classifier.py) · [כלי התיוג](human_review.py) · [מחלץ המקורות](../compass_v9_14/decision_miner.py).
