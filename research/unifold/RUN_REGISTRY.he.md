# UNIFOLD — רישום קבוע של ריצות GitHub Actions
**נבדק מול GitHub Actions בזמן אמת:** 2026-10-12  
**מאגר:** [toyro396133/personal-native-lm-5m](https://github.com/toyro396133/personal-native-lm-5m)  
**ענף:** `experiment/unifold-foundations`  
**PR:** [#5](https://github.com/toyro396133/personal-native-lm-5m/pull/5)

## ריצות תיקון שאומתו בפועל
| Run ID | Workflow | Commit SHA | Job ID | תוצאה מאומתת | Artifact |
|---|---|---|---|---|---|
| [38228625518](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38228625518) | Gate 0/1 — עם צמצום טריגרים | `4f1cb7c1c0a3af6624968f1e9557094376a17027` | `114742275274` | **SUCCESS** — תוקן גם סינון הטריגרים | `unifold-gate0-validation` — `11738155585` |
| [38228435185](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38228435185) | Gate 0/1 datasets | `88b6eff082e67d4864943f2d601d67d8cf0738fe` | `114741620126` | **SUCCESS** — 10/10 בדיקות יחידה, 10,000 משימות ו־2,000 רצפים עברו ביקורת מלאה | `unifold-gate0-validation` — `11738090381` |
| [38228435176](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38228435176) | Read-only Actions audit | `88b6eff082e67d4864943f2d601d67d8cf0738fe` | `114741619727` | **SUCCESS** — דוח ריצות ב-Job Summary ובארטיפקט, ללא כתיבה ל-PR | `unifold-run-audit` — `11737975783` |

**מסקנה:** שלושת הכישלונות שנמסרו נגרמו מתקלות הגדרות CI/הרשאות; שני השינויים הוכחו במפורש ב־GitHub Actions על commit `88b6eff`. זו אינה הוכחת הצלחה מדעית של מנגנון Q.

## חמש הריצות שנמסרו ונבדקו
| Run ID | Workflow | Commit SHA | Job ID | סיכום | Artifact |
|---|---|---|---|---|---|
| [38225769530](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38225769530) | Neural M/Q pilot | `75152555bca6afdb5b30b62ec14374f18f776c76` | `114732776600` | GitHub SUCCESS; אין ראיה ליתרון Q | `unifold-neural-pilot` — `11736141509` |
| [38225887901](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38225887901) | Support learning calibration | `5624bfd07af202b1794f8d08f738c6610b743127` | `114733169337` | GitHub SUCCESS; עקומת למידה במשפחה פשוטה | `unifold-support-learning-calibration` — `11736246640` |
| [38226032657](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38226032657) | Dataset gate 0/1 | `75499e51595522b1275b63801a70e57a4a7107d9` | `114733646216` | FAILURE: ModuleNotFoundError: No module named 'torch' בעת גילוי כל בדיקות הנתונים | ללא |
| [38226126730](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38226126730) | Dataset gate 0/1 | `07b19d9f97c64201d2a3be2bc8bb3666c72ad2fe` | `114733955720` | FAILURE: אותה תקלה — בדיקות מודל נטענו ב-Workflow בלי PyTorch | ללא |
| [38227458399](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38227458399) | Runs audit | `86db59ef32f3d6bd24109fe38540f46de914f9fd` | `114738367075` | FAILURE: קריאת ההרצות הצליחה; פרסום תגובה ב-PR החזיר HTTP 403 `Resource not accessible by integration` | ללא |

## ראיות מדעיות מהלוגים של GitHub
### M/Q — ריצה 38225769530
- 66,944 פרמטרים, 120 צעדי אימון, 8 זוגות בכל צעד, train=150 זוגות, val=20 זוגות.
- דיוק **סמל בודד** בזרוע linked לאחר 12 דוגמאות: B ברמה L1 עם Q משתנה `42.2656%`; Q קפוא `42.1875%`. C ברמה L2 עם Q משתנה `37.3437%`; Q קפוא `37.5000%`.
- עקומת B עם Q משתנה (1 → 12 דוגמאות): `42.50% → 42.2656%`. C: `36.9531% → 37.3437%`.
- **פסק דין:** SMOKE SUCCESS בלבד; אין הוכחת למידת חוקיות אפיזודית ב-L1/L2 ואין שיפור אמין באמצעות Q. אי אפשר להשוות ישירות למדדי משפחת כיול אחרת.

### M-only ו-positive control — ריצה 38225887901
- מודל M-only עצבי עם **21,253 פרמטרים**. 1,200 צעדי אימון, מבחן 4,000 אפיזודות. מבחן סמל יחיד עם 5 תוצאות אפשריות (ניחוש: 20%).
- דיוק M-only עם 0, 1, 2, 4, 8, 12 דוגמאות: **19.8%, 24.975%, 31.35%, 45.425%, 66.875%, 80.075%**.
- עם 12 פלטי תמיכה מעורבבים: **20.325%**. זה מראה שימוש ממשי בדוגמאות בתוך משפחת חוקים אפינית סגורה.
- positive control שמקבל מראש היפותזות אפיניות (81 פרמטרים): `87.075%` עם 12 דוגמאות, `20.85%` עם פלטים מעורבבים.
- **פסק דין:** PASS על למידה *בתוך משימה* פשוטה בלבד; לא הוכחה צמיחת יכולת למידה בין משימות או L2.

## למה שלוש הריצות נכשלו וכיצד תוקנו בקוד
- Gate 0/1 (שני כישלונות): `unifold-dataset-gate.yml` השתמש ב-`unittest discover` על **כל** הבדיקות, כולל `test_learner.py` ו-`test_support_diagnostic.py`, שדורשות `torch`. ה-Workflow לא התקין אותו. התיקון מגביל את הבדיקות ל-`test_taskgen.py` ו-`test_curriculum.py` — בדיקות dataset ללא תלות ב-ML.
- Runs audit: ל-GITHUB_TOKEN אין הרשאת כתיבת תגובות ל-PR בענף/הקשר הנוכחי, ולכן פרסום דרך `issues.createComment` נכשל ב-403. התיקון מבטל **כל ניסיון כתיבה ל-PR**, וכותב דוח ל-GitHub Actions Job Summary ולארטיפקט טקסט לקריאה חוזרת.
- תיקונים אלה מחייבים **ריצת CI חדשה**; אין לטעון שהכישלונות הישנים תוקנו בפועל עד שיש Run ID חדש עם SUCCESS.

## הקשר: הרצת Gate 0/1 קודמת שהצליחה
[38222537701](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38222537701) — גירסת נתונים קודמת, SUCCESS לפי רישום ה-GitHub audit. עצם הצלחת גרסה קודמת אינה מחליפה בדיקה בגרסת הקוד המעודכנת.

## נוהל קבוע להמשך
לכל Workflow שמפעילים או מזהים **לשמור מייד**: `run_id`, כתובת הריצה, `job_id`, שם ה-Workflow, branch, `head_sha`, attempt, תצורה/seed, סטטוס, תוצאות מדודות, `artifact_id`, החלטות והמשך. אין לשייך משקלים או תוצאות של SELF/COMPASS ל-UNIFOLD. אין לכנות `CI success` הצלחה מדעית.

**הערה:** התוצאות כאן נקראו ישירות מלוגי GitHub Actions של הריצות המסוימות, ולא רק מדוחות הרצות מקומיות.


## מדיניות טריגרים (12.10.2026)
מאגר UNIFOLD חולק תשתית עם SELF, ולכן Workflow בדיקות הנתונים מאז תיקון זה מופעל רק בשינוי `taskgen.py`, `curriculum.py`, בדיקותיהם או קובץ ה-Workflow עצמו; **שינויי תיעוד אינם משיקים בדיקות חדשות**. נמשיך לשמור כאן Run IDs, SHAs והמדדים של הרצות חדשות מבלי לעורר מחדש אימונים או בדיקות שלא לצורך.
