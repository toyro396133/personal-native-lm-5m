# מאגר הנתונים LFM3-Cyber — Pilot v2

## מצב המאגר
המאגר כולל **200 דוגמאות סינתטיות מקוריות** בשמונה תחומים. החלוקה היא 160 Train, 20 Validation ו־20 Test (80/10/10). המאגר עדיין אינו מאגר מאושר לאימון: כל 200 הדוגמאות מסומנות `pending` וממתינות לביקורת מומחה אנושית. בדיקות אוטומטיות מאמתות מבנה, שדות, מזהים, כפילויות ובידוד splits — לא נכונות מקצועית מלאה.

## קבצים
- קובצי הבסיס: `pilot/train.jsonl`, `pilot/validation.jsonl`, `pilot/test.jsonl`.
- קובצי הרחבה: `pilot/train-2.jsonl` עד `pilot/train-5.jsonl`, `pilot/validation-2.jsonl`, `pilot/test-2.jsonl`.
- כל סקריפטי הטעינה מקבלים תיקייה וקוראים את כל קובצי `*.jsonl` שבה; `prepare_dataset.py` מאחד אותם ל־3 קובצי פלט סופיים לפי split.
- `schema.json`: JSON Schema Draft 2020-12.
- `sources/source-registry.json`: מקורות, כתובות, רישוי ומדיניות שימוש.
- `reports/*.json`: דוחות כיסוי, איכות ופיצול.
- `reports/quality-review-2026-11-06.md`: ביקורת איכות מפורטת, ממצאים ותיקונים.
- `../scripts/validate_dataset.py`: בדיקת JSONL מול JSON Schema בפועל, מזהים, הרשאות, מקור, כפילויות ודליפת split.
- `../scripts/deduplicate_dataset.py`: איתור כפילויות מדויקות ודמיון טקסטואלי.
- `../scripts/audit_dataset.py`: יצירת דוחות כיסוי, איכות ופיצול.

## סכמת רשומה
שדות חובה: `id`, `scenario_id`, `split`, `domain`, `task`, `environment`, `difficulty`, `orientation`, `messages`, `expected_skills`, `verification`, `source`, `quality`, `authorization`, `provenance`.
ערכי enum: domain בשמונה התחומים המוגדרים בסכמה; split הוא train/validation/test; environment הוא defensive/authorized_lab/conceptual; difficulty הוא beginner/intermediate/advanced; verification.status הוא pending/automated/expert_review/verified; quality.review_status הוא pending/reviewed/rejected.
שדות מקור ורישוי הם חובה גם לדוגמאות סינתטיות; במקרה כזה מציינים במפורש מקור יצירה מקורי ורישיון מתאים. אין למלא מקור חיצוני שלא שימש בפועל. `url` ריק מותר רק למקור יצירה מקורי.

## כללי איכות ומקור
- התוכן נכתב כדוגמאות מקוריות; המקורות משמשים מסגרת מקצועית ולא טקסט להעתקה. סטטוס הרישוי מתועד לכל מקור ב-source registry; במקורות ממשלתיים נותרת בדיקת הודעות צד שלישי ברמת המסמך הספציפי.
- אין לייחס למקור רישיון CC-BY-4.0 באופן גורף. ATT&CK כפוף לתנאי MITRE והנחיות סימן מסחר; NIST/CISA הם בדרך כלל יצירות ממשלת ארה"ב אך יש לבדוק חריגי צד שלישי; Sigma specification שונה מכללי SigmaHQ (DRL 1.1); OWASP Cheat Sheet Series תחת CC-BY-SA-4.0.
- בדוגמאות authorized_lab היעד מוגבל למכונת/רשת מעבדה שהוקצו במפורש. אין דוגמאות לניצול מערכות צד שלישי, גניבת מידע, נוזקה, התמדה או התחמקות זדונית.
- אין לטעון שהדוגמה נבדקה ידנית. כל הדוגמאות הנוכחיות ממתינות לביקורת מומחה.
- יש להוסיף תאריך/גרסה עבור מידע תלוי-זמן. אין להכניס IOC או CVE דינמיים ללא אימות מקור וזמן.

## פיצול למניעת דליפה
החלוקה היא 160 Train, 20 Validation ו־20 Test (80/10/10), לפי מזהי תרחיש ייחודיים. כל קובצי ה־JSONL בתיקייה נטענים יחד לפני הבדיקה וההכנה. ה-validator מונע scenario_id שחוצה splits, תוכן זהה בין splits ודמיון טקסטואלי גבוה בין prompt/answer ב-splits שונים. בעת הרחבה, כל וריאציה של תרחיש חייבת לשמור אותו scenario_id ולהישאר באותו split. Test הוא holdout ואסור להשתמש בו לאימון או לכוונון.

## הרצה
```bash
python projects/lfm3-cyber/scripts/validate_dataset.py projects/lfm3-cyber/dataset/pilot
python projects/lfm3-cyber/scripts/deduplicate_dataset.py projects/lfm3-cyber/dataset/pilot
python projects/lfm3-cyber/scripts/audit_dataset.py projects/lfm3-cyber/dataset/pilot --output-dir projects/lfm3-cyber/dataset/reports
python -m unittest discover -s projects/lfm3-cyber/tests -v
python projects/lfm3-cyber/scripts/prepare_dataset.py projects/lfm3-cyber/dataset/pilot --output-dir projects/lfm3-cyber/data/prepared
```

## מעבר ל-500
נותרו 300 דוגמאות נוספות ליעד 500. לפני קידום לאימון יש לבצע ביקורת מומחה, להרחיב גיוון בפורמט (לוגים סינתטיים, תצורות, timelines, multi-turn וכתיבת דוחות), להוסיף מטריצת תתי־מיומנויות ולדייק הפניות למסמכי מקור ספציפיים. אין לקדם דוגמאות ל־`verified` ללא בודק אנושי וראיות לאימות.
