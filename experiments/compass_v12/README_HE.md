# COMPASS V12 — שכבה עצבית חיצונית שמתאמנת למשתמש

**מסלול COMPASS עצמאי. SELF לא שונה.** מבוסס על רעיון המשתמש: שכבה קטנה, בעלת משקלים נלמדים, שנשארת מחוץ למודל הגדול ומתאמנת בכל פעם שיש משוב אמין. המודל הגדול נשאר קפוא וממשיך לבצע את החשיבה.

## מה באמת קיים ב־V12

**1. `FrozenResidualBridge`:** מתאם Transformer בצורת V9.0: היטל `hidden → rank → hidden` בתוספת מצב אישי ומצב פרויקט. המתאם מוכנס באמצעות Hook אמיתי לפלט הבלוק האחרון של מודל שפה גדול. אפשר לטעון לתוכו את `down.weight` ו־`up.weight` מקובץ V9.0 המאומן הישן. הוא **אינו** משתמש באופן שגוי במשקלי דירוג V8.5, שבנויים למשימה אחרת.

**2. `PersonalProjectState`:** לכל משתמש וקטור אישי; לכל זוג משתמש–פרויקט וקטור פרויקטלי נוסף. שניהם **מחוץ למשקלי המודל הגדול ומחוץ למשקלי המתאם**. אפשר לשמור אותם בקובץ JSON פרטי, לטעון מחדש, ולעבור למודל אחר כשיהיה מתאם מתאים.

**3. `personalized_update`:** מאמן רק את הווקטור ששייך למשוב הנוכחי. המודל הגדול והמתאם קפואים. בודק משוב משתמש מפורש עם מקור, מעדכן באמצעות גרדיאנטים, שומר תיעוד ומונע אימון כפול של אותו אירוע.

**4. `from_v11_feedback.py`:** ממיר תיקונים מקוריים שאומתו במסגרת V11.1 לזוגות **אימון** ל־V12 — ההצעה שנדחתה מול התיקון שהמשתמש אישר. לא מכנה אותם דוגמאות מבחן עתידיות, משום שהתשובה כבר ידועה בעת יצירת הזוג.

## בדיקות שהורצו בפועל

- [בדיקות אינטגרציה על Transformer קטן](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38190008600): **18 בדיקות עברו**. אימנו את וקטור הפרויקט והפעלנו חיבור עצבי בתוך המודל. בדקנו בידוד משתמשים, בידוד פרויקטים, שמירה וטעינת מצב, מניעת אישור מדיניות ובקרת מקור.
- [בדיקת SmolLM2-360M אמיתית, מתאם מאותחל אקראית](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38190117236): **SUCCESS הנדסי**. נורמת שינוי וקטור הפרויקט `0.000222726`; מרווחי Yes−No לפני עדכון `[0.2847433,0.9049158]` ולאחריו `[0.2847424,0.9049129]`; הבחירה לא השתנתה. המשמעות: גרדיאנטים עוברים בתוך המודל, אך השפעת האימון האישי עם מתאם אקראי **זעירה ולא מועילה במדד הזה**.
- [בדיקות צנרת משוב V11→V12](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38190351282): נוספו בהמשך, אין לקרוא להן הצלחה עד שתוצאות הריצה ייבדקו.
- [הפעלה נוספת: מתאם V9.0 המאומן ההיסטורי על אותו SmolLM2-360M](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38190341185): בודקת שימוש במשקלים המאומנים הקיימים במקום מתאם אקראי. התוצאה תירשם בדוח נפרד לאחר בדיקה.

### ניסוי ביקורת מותאם ובדיקת V9.0

ב־[ביקורת 360M מותאמת עם מתאם אקראי](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38190627871), אותן הגדרות של 10 צעדי אימון ו־lr=0.4 הניבו שינוי מצב בגודל **0.000741** ושינוי לא מועיל במרווח העדפה (כ־**0.620163 → 0.620162**). ב־[מתאם V9.0 המאומן](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38190341185) נמדד **0.189108** ושיפור במרווח ההעדפה (**0.315979 → 0.337484**). זהו יתרון מכני במקרה סינתטי אחד, **לא** ראיה ליכולת הבנה כללית; למתאמים נקודות פתיחה שונות.

### PowerShell — הרצת הוכחת החיבור העצבי

ללא GPU, על Windows ייתכן שהבדיקה תהיה איטית ותדרוש כמה ג׳יגה־בייט פנויים. GitHub Actions כבר משמש להרצת המעבד החיצוני.

```powershell
git fetch origin
git switch experiment/compass-v12-detachable-personal-layer
python -m pip install torch transformers safetensors sentencepiece
$env:PYTHONPATH = "experiments/compass_v12"
python experiments/compass_v12/smoke_360m_optional.py `
  --report "$env:USERPROFILE\Documents\compass-v12-report.json" `
  --state "$env:USERPROFILE\Documents\compass-v12-SYNTHETIC-state.json"
```

לניסוי מתאם V9.0 המאומן, הוסף `--v90-checkpoint "path\to\compass-v90-personal-neural.safetensors"`. אין צורך לאמן מחדש את המודל הגדול. **משקלי המתאם המאומנים ב־V9.0 משותפים, ולא משקלים ייחודיים למשתמש.** החידוש של V12 הוא שהווקטורים האישיים והפרויקטליים נלמדים בכל סבב משוב ונשמרים בנפרד.

### מסלול העבודה השלם: ללמוד שוב ושוב ממשוב שאושר

אחרי יצירת קובץ `compass-v12-training-pairs.json` מהסקירה של V11.1, אפשר **לאמן את השכבה האישית בפועל דרך מודל גדול קפוא**:

```powershell
$env:PYTHONPATH = "experiments/compass_v12"
python experiments/compass_v12/smoke_360m_optional.py `
  --v90-checkpoint "C:\path\to\compass-v90-personal-neural.safetensors" `
  --training-pairs "$env:USERPROFILE\Documents\CompassPrivate\compass-v12-training-pairs.json" `
  --report "$env:USERPROFILE\Documents\CompassPrivate\compass-v12-train-report.json" `
  --state "$env:USERPROFILE\Documents\CompassPrivate\compass-v12-personal-state.json"
```

בפעם הבאה שמתקבל משוב נוסף, מאשרים אותו ויוצרים **קובץ חדש המכיל רק אירועים שלא אומנו כבר**; מריצים שוב עם `--existing-state` המצביע לאותו קובץ מצב קודם. המערכת חוסמת אירוע חוזר, לא מערבבת בין פרויקטים ומשאירה את המודל הגדול ואת המתאם המשותף קפואים.

**אזהרות:** ב־Windows עם מחשב חלש הרצת 360M יכולה להיות איטית. עדיף להריץ בסביבת מחשוב מתאימה. אין להעלות את קובצי השיחות/האישיות למאגר GitHub ציבורי או לשתף אותם כחלק מ־Actions ציבורי. גיבוי ושמירת הרשאות הקבצים הם באחריות סביבת העבודה.

### הפיכת משוב V11.1 לקובץ אימון V12 — ללא מודל חיצוני

```powershell
$env:PYTHONPATH = "experiments/compass_v12"
python experiments/compass_v12/from_v11_feedback.py `
  --events "$env:USERPROFILE\Documents\CompassPrivate\compass-v111-private-chat-events.jsonl" `
  --reviews "$env:USERPROFILE\Downloads\compass-v11-1-private-human-reviews.json" `
  --user-id "my-user" `
  --output "$env:USERPROFILE\Documents\CompassPrivate\compass-v12-training-pairs.json"
```

הקובץ הוא **תור דוגמאות אימון פרטיות**, לא הוכחה שמדיניות פרויקט אושרה ולא מבחן הסקה עתידי. מזהי ההודעות משמשים לקישור למקור; המערכת עדיין אינה מבצעת אימות זהות קריפטוגרפי.

## מה עדיין אין

- אין הוכחה שהשכבה האישית שיפרה הבנת חזון או החלטות אמיתיות של משתמש לאורך זמן.
- אין עדיין מתאם מאומן זמין לכל משפחת מודלים; מעבר בין מודלים עדיין לא נבדק.
- אין עדיין מנגנון אוטונומי אמין להפיק גרדיאנטים מהמשמעות של כל תשובה חופשית — V11.1 מספק אישורי תיקון ידניים.
- אין שום חיבור נדרש ל־SELF; הוא מפותח בנפרד.

**פסק דין בשלב זה:** חיבור עצבי ולמידת מצב אישי עובדים הנדסית; השיפור המהותי האישי עוד לא הוכח.

[פרוטוקול V12](PROTOCOL.md) · [קוד שכבת COMPASS](detachable_layer.py) · [הרצת 360M](smoke_360m_optional.py) · [קוד יבוא המשוב](from_v11_feedback.py).
