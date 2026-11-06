# מאגר הנתונים LFM3-Cyber — Pilot v1

## מצב המאגר
זהו Pilot ראשוני של **40 דוגמאות סינתטיות מקוריות** בשמונה תחומים. הוא אינו עומד עדיין ביעד 500 ואינו מאגר שעבר אימות מומחה. כל הדוגמאות מסומנות `pending` עד לביקורת אנושית. הבדיקות האוטומטיות מאמתות מבנה, שדות, מזהים, כפילויות ודליפת split בלבד — לא נכונות מקצועית.

## קבצים
- `pilot/train.jsonl`, `pilot/validation.jsonl`, `pilot/test.jsonl`: פיצול 80/10/10 לפי קבוצות תרחיש.
- `schema.json`: JSON Schema Draft 2020-12.
- `sources/source-registry.json`: מקורות, כתובות, רישוי ומדיניות שימוש.
- `reports/*.json`: דוחות כיסוי, איכות ופיצול.
- `../scripts/validate_dataset.py`: בדיקת JSONL, schema, מזהים, הרשאות, כפילויות ודליפת split.
- `../scripts/deduplicate_dataset.py`: איתור כפילויות מדויקות ודמיון טקסטואלי.
- `../scripts/audit_dataset.py`: יצירת דוחות כיסוי, איכות ופיצול.

## סכמת רשומה
שדות חובה: `id`, `scenario_id`, `split`, `domain`, `task`, `environment`, `difficulty`, `messages`, `expected_skills`, `verification`, `source`, `quality`, `authorization`, `provenance`.
ערכי enum: domain בשמונה התחומים המוגדרים בסכמה; split הוא train/validation/test; environment הוא defensive/authorized_lab/conceptual; difficulty הוא beginner/intermediate/advanced; verification.status הוא pending/automated/expert_review/verified; quality.review_status הוא pending/reviewed/rejected.
שדות מקור ורישוי הם חובה גם לדוגמאות סינתטיות; במקרה כזה מציינים במפורש מקור יצירה מקורי ורישיון מתאים. אין למלא מקור חיצוני שלא שימש בפועל. `url` ריק מותר רק למקור יצירה מקורי.

## כללי איכות ומקור
- התוכן נכתב מחדש כדוגמאות מקוריות; המקורות משמשים מסגרת מקצועית ולא טקסט להעתקה.
- אין לייחס למקור רישיון CC-BY-4.0 באופן גורף. ATT&CK כפוף לתנאי MITRE והנחיות סימן מסחר; NIST/CISA הם בדרך כלל יצירות ממשלת ארה"ב אך יש לבדוק חריגי צד שלישי; Sigma specification שונה מכללי SigmaHQ (DRL 1.1); OWASP Cheat Sheet Series תחת CC-BY-SA-4.0.
- בדוגמאות authorized_lab היעד מוגבל למכונת/רשת מעבדה שהוקצו במפורש. אין דוגמאות לניצול מערכות צד שלישי, גניבת מידע, נוזקה, התמדה או התחמקות זדונית.
- אין לטעון שהדוגמה נבדקה ידנית. כל הדוגמאות הנוכחיות ממתינות לביקורת מומחה.
- יש להוסיף תאריך/גרסה עבור מידע תלוי-זמן. אין להכניס IOC או CVE דינמיים ללא אימות מקור וזמן.

## פיצול למניעת דליפה
החלוקה נעשתה לפי מזהי תרחיש ייחודיים, ולא באמצעות ערבוב אקראי של שורות. ה-validator מונע scenario_id שחוצה splits, תוכן זהה בין splits ודמיון טקסטואלי גבוה בין prompt/answer ב-splits שונים. בעת הרחבה, כל וריאציה של תרחיש חייבת לשמור אותו scenario_id ולהישאר באותו split. Test הוא holdout ואסור להשתמש בו ליצירת דוגמאות או לכוונון.

## הרצה
```bash
python projects/lfm3-cyber/scripts/validate_dataset.py projects/lfm3-cyber/dataset/pilot
python projects/lfm3-cyber/scripts/deduplicate_dataset.py projects/lfm3-cyber/dataset/pilot
python projects/lfm3-cyber/scripts/audit_dataset.py projects/lfm3-cyber/dataset/pilot --output-dir projects/lfm3-cyber/dataset/reports
python -m unittest discover -s projects/lfm3-cyber/tests -v
```

## מעבר ל-500
יש להשלים 460 דוגמאות נוספות, לאחר סקירת מומחה מדגמית/מלאה, להגדיל גיוון בתתי-תחומים, להוסיף דוגמאות עם לוגים ונתוני תצורה סינתטיים, ולהפעיל בדיקות רישוי לכל מקור חדש. אין לקדם דוגמאות ל-`verified` ללא מתעד/בודק אנושי וראיות לאימות.
