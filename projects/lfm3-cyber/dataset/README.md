# מאגר הנתונים LFM3-Cyber — Pilot v3

## מצב המאגר
המאגר כולל **1,100 דוגמאות סינתטיות מקוריות** בשמונה תחומים: 890 Train, 110 Validation ו־110 Test (80/10/10). כל 1,100 הדוגמאות עדיין מסומנות `pending` וממתינות לביקורת מומחה אנושית. בדיקות אוטומטיות מאמתות מבנה, סכימה, מזהים, כפילויות, בידוד splits וכיסוי מיומנויות — לא נכונות מקצועית מלאה.

## קבצים ומבנה
- קובצי הבסיס: `pilot/train.jsonl`, `pilot/validation.jsonl`, `pilot/test.jsonl`.
- הרחבות אימון: `train-2.jsonl` עד `train-11.jsonl`.
- הרחבות Validation: `validation-2.jsonl` עד `validation-7.jsonl`.
- הרחבות Test: `test-2.jsonl` עד `test-7.jsonl`.
- כל קובצי `*.jsonl` בתיקיית Pilot נטענים יחד; `prepare_dataset.py` מאחד אותם לפי split.
- `schema.json`: JSON Schema Draft 2020-12.
- `skill-matrix.json`: מטריצת 60 מיומנויות יעד בשמונה תחומים, עם רף מינימלי של חמש דוגמאות לכל מיומנות.
- `sources/source-registry.json`: מקורות, רישיונות ומדיניות שימוש.
- `reports/skill-coverage-report.json`: ספירת כיסוי בפועל לפי תחום ומיומנות.
- `reports/quality-review-2026-11-06.md`: ממצאי ביקורת איכות והרחבות.
- `reports/expansion-1000-2026-11-06.md`: תיעוד הרחבת המאגר ל־1,000 דוגמאות.
- `reports/ai-audit-2026-11-06.md`: ביקורת AI ראשונית על גרסת 500 הרשומות; הוחלפה בהרחבה המאוחרת לדוח 1,000 הרשומות, כולל ממצאים והמלצות לתיקון.
- `../scripts/validate_dataset.py`: בדיקת רשומות מול JSON Schema, מזהים, הרשאות, מקור, כפילויות ודליפת split.
- `../scripts/deduplicate_dataset.py`: איתור כפילויות מדויקות ודמיון טקסטואלי בין splits.
- `../scripts/audit_dataset.py`: דוחות כיסוי, איכות ופיצול.
- `../scripts/audit_skill_coverage.py`: בדיקת מטריצת תתי־מיומנויות.

## גיוון פורמטים
ההרחבות הוסיפו 122 קבוצות תרחיש עצמאיות חדשות: 40 קבוצות ראשונות, 12 קבוצות נוספות ו־48 קבוצות נוספות, כל אחת בחמש צורות משימה:
1. ניתוח מקרה ותעדוף פעולות.
2. ניתוח ארטיפקט סינתטי (לוג/תצורה/נתוני אירוע).
3. שיחה רב־שלבית עם שאלת הבהרה.
4. ניתוח ציר זמן עם אי־ודאות בזמנים.
5. כתיבת דוח מובנה עם תצפיות, הערכה, פעולות ולא־ידוע.

חמש הווריאציות של כל תרחיש חולקות `scenario_id` ונשארות באותו split. הארטיפקטים סינתטיים ואינם טלמטריה אמיתית. הגיוון משפר את צורות הקלט והפלט, אך עדיין נדרשת הרחבה נוספת לתרגול קוד, שאילתות, תצורות מלאות, ניתוח PCAP סינתטי ותרגילי multi-turn מורכבים יותר.

## מטריצת מיומנויות
המטריצה מגדירה 60 מיומנויות עדיפות בשמונה תחומים. הסקריפט סופר רק תיוג מפורש ב-`expected_skills`, בתוך התחום המתאים. רף של חמש דוגמאות הוא רף כיסוי מינימלי בלבד, לא הוכחת שליטה או איכות. בדיקת CI נכשלת אם אחת ממיומנויות היעד אינה מגיעה לרף.

## כללי איכות ומקור
- התוכן נכתב כדוגמאות מקוריות; המקורות משמשים מסגרת מקצועית ולא טקסט להעתקה.
- אין לייחס למקור רישיון CC-BY-4.0 באופן גורף. ATT&CK כפוף לתנאי MITRE והנחיות סימן מסחר; NIST/CISA הם בדרך כלל יצירות ממשלת ארה"ב אך יש לבדוק חריגי צד שלישי; Sigma specification שונה מכללי SigmaHQ (DRL 1.1); OWASP Cheat Sheet Series תחת CC-BY-SA-4.0.
- בדוגמאות authorized_lab היעד מוגבל למכונת/רשת מעבדה שהוקצו במפורש. אין דוגמאות לניצול מערכות צד שלישי, גניבת מידע, נוזקה, התמדה או התחמקות זדונית.
- אין לטעון שהדוגמה נבדקה ידנית. כל 1,000 הדוגמאות ממתינות לביקורת מומחה.
- יש להוסיף תאריך/גרסה למידע תלוי־זמן. אין להכניס IOC או CVE דינמיים ללא אימות מקור וזמן.

## פיצול למניעת דליפה
החלוקה היא 890 Train, 110 Validation ו־110 Test (80/10/10), לפי קבוצות תרחיש. כל הווריאציות של תרחיש נשארות באותו split. ה-validator בודק מזהים, scenario_id, תוכן זהה ודמיון טקסטואלי גבוה בין splits. Test הוא holdout ואסור להשתמש בו לאימון או לכוונון.

## הרצה
```bash
python projects/lfm3-cyber/scripts/validate_dataset.py projects/lfm3-cyber/dataset/pilot
python projects/lfm3-cyber/scripts/deduplicate_dataset.py projects/lfm3-cyber/dataset/pilot
python projects/lfm3-cyber/scripts/audit_dataset.py projects/lfm3-cyber/dataset/pilot --output-dir projects/lfm3-cyber/dataset/reports
python projects/lfm3-cyber/scripts/audit_skill_coverage.py projects/lfm3-cyber/dataset/pilot --output projects/lfm3-cyber/dataset/reports/skill-coverage-report.json
python -m unittest discover -s projects/lfm3-cyber/tests -v
python projects/lfm3-cyber/scripts/prepare_dataset.py projects/lfm3-cyber/dataset/pilot --output-dir projects/lfm3-cyber/data/prepared
```

## שער לפני אימון
אין להתחיל SFT עד שכל 1,100 הדוגמאות עברו ביקורת מומחה מדגמית/מלאה לפי תוכנית QA מאושרת, הפניות המקור ברמת מסמך, בדיקות הכיסוי והסכמה עברו, ונבדקה איכות התשובות מול מקורות סמכותיים. אין לקדם רשומה ל-`verified` ללא בודק אנושי וראיות לאימות.
