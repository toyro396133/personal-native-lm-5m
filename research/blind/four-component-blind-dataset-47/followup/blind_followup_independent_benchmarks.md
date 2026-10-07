# Blind Follow-up Independent Benchmarks

**מחקר המשך עיוור לחלוטין**  
Repository: `toyro396133/personal-native-lm-5m`  
מקור נתונים עיוור: GitHub Actions run `37694378705`, artifact `four-component-blind-dataset-47`  

> לא בוצע unblinding. לא נקראו README, תיעוד הפרויקט, `EXPERIMENT_LOG.md`, קוד ניסויים, workflows ישנים, commit history, שמות checkpoints אמיתיים או מיפוי של A/B/C/D ו־V1–V7.

---

## תקציר מנהלים

השלב השני נועד **לנסות להפריך** את המפה שנבנתה בשלב הראשון, ולא לחפש אישור נוח. לשם כך קיבעתי תחילה את ההשערות ואת תוכנית הניסויים ב־Git לפני צפייה בתוצאות חדשות:

- preregistration ראשון: `67f6c12e923043066ae57b1f83d06a9286f4d78e`
- artifact-only addendum: `2125b5ac26c5dc93a2f5e745c588dc0151969d96`

**Preregistration integrity note:** the first commit correctly preserved the hypothesis snapshot, but its intended causal `benchmark_plan.md` was accidentally replaced by a file-bridge error placeholder. The five F1–F5 benchmarks that actually produced results were fully specified in the second commit before execution. The intended causal plan is restored in the final results package from the local copy whose SHA-256 was recorded before execution; this restoration is not presented as retroactive preregistration.

התוכנית המקורית כללה activation patching, magnitude-matched interventions על A/B/C/D, donor substitution ו־counterfactual composition. בפועל לא נמצא במאגר או ב־run העיוור manifest אנונימי שמאפשר גישה למשקלים כ־V1–V7 בלי לעבור דרך שמות או קוד שעלולים לבצע unblinding. לכן **לא זייפתי ריצה סיבתית שלא התבצעה**. במקום זאת ננעל addendum לפני הריצה עם חמישה falsification benchmarks חדשים המבוססים אך ורק על primitive blind measurements שכבר קיימים ב־artifact.

התוצאה המרכזית היא שהמפה הקודמת השתנתה בכמה נקודות חשובות:

1. **A עדיין נראה כרכיב דינמי בעל השפעה ממשית, אבל רוב הנזק הגולמי של `negate` מוסבר על ידי גודל ההפרעה, לא על ידי הסימן לבדו.** לאחר תיקון ל־effect norm נשאר אפקט mode קטן בלבד, ובכיוון הפוך מן הסיפור הפשוט: `negate` פחות מזיק ממה שהעוצמה שלו לבדה הייתה מנבאת.
2. **הערך הנלמד של A אינו reference אוניברסלי.** עבור C אין יתרון pooled כמעט בכלל, אבל יש polarity חזקה לפי וריאנט: V5 שלילי ב־42/42 שכבות, V4 חיובי ב־40/42, V6 חיובי ב־41/42, V3 שלילי ב־37/42.
3. **הסיפור הפשוט של “פרמטר מנגנון בשכבה L מתחזק ואז גורם ל־A effect בשכבה L+1” נכשל.** lag regression נותן `R²=0.0186`, עם permutation `p=0.628`.
4. **D הוא challenge עקבי הרבה יותר מ־C ביחס ל־B.** על פני 294 רשומות שכבה בכיסוי המלא, `B_under_D - B_under_C` היה שלילי ב־198, אפס ב־96, וחיובי ב־0. כל שבעת הווריאנטים מראים ממוצע שלילי.
5. **אין fingerprint יחיד שמתייצב מונוטונית לאורך האימון.** intervention fingerprint ב־25M כבר דומה ל־120M, אבל ב־50M הדמיון נעלם. לעומת זאת reference fingerprint יציב במיוחד בין 50–110M ואז משתנה חלקית ב־120M. זה נראה יותר כמו כמה clocks התפתחותיים מאשר שלב יחיד שבו “הארכיטקטורה מתקבעת”.

---

# 1. ההשערות המקוריות שקובעו מראש

## A

**Observed behavior מהשלב הראשון:** כיוון עקבי מוקדם, effect שמתחזק בשכבות מאוחרות יותר, והפרדה בין strength, directional consistency ו־learned-value specificity.

**Frozen interpretation:** A הוא ציר דינמי של control/reference/routing. עצם קיומו ועוצמתו יכולים להיות חשובים גם כאשר הערך הספציפי שנלמד אינו מועיל.

**Confidence לפני ההמשך:** בינוני־גבוה לגבי “ציר דינמי מובחן”; בינוני לגבי “reference/control”.

## B

**Frozen interpretation:** B הוא עוגן יציב יחסית, identity/content-like, שאינו תלוי במנגנון A המדיד כדי להתקיים.

**Confidence לפני ההמשך:** גבוה לגבי יציבות, בינוני לגבי “anchor” במובן סיבתי.

## C

**Frozen interpretation:** C הוא רכיב יחסי/קונטקסטואלי שהביטוי שלו תלוי בחלקו ביחס ל־B.

**Confidence לפני ההמשך:** בינוני.

## D

**Frozen interpretation:** D הוא משתנה מאתגר/מתחרה או ממד context שמפריע ל־B באופן רחב יותר מ־C.

**Confidence לפני ההמשך:** נמוך־בינוני.

## יחסים ועומק

ההשערה הקפואה הייתה:

- B הוא היציב ביותר.
- A ו־B עשויים להיות שני “עוגנים” מסוגים שונים: B יציב, A דינמי.
- A אינו הכרחי לעצם קיומו של B.
- C יחסי יותר ל־B מאשר D.
- D עשוי להיות entangled/competitive יותר.
- סדר העומק המשוער: כיוון A מוקדם -> compression סביב layer 2 -> recovery/reorganization בשכבות 3–4 -> A effect חזק סביב 5–6 -> אובדן חלק מן הגאומטריה ב־final.

## משפחות וריאנטים לפני ההמשך

- V1: distinct/control-like.
- V2–V3: רגישים יחסית ל־A.
- V4: רגיש ל־A אך עם learned-reference חיובי יותר.
- V5: A effect חזק אך learned-reference ל־C שלילי/בעייתי.
- V6–V7: B עמיד יחסית להתערבויות A.

---

# 2. אילו benchmarks תוכננו ולמה

## התוכנית הסיבתית המקורית

נרשמו מראש חמישה benchmark families:

1. magnitude- and angle-matched component intervention matrix על A/B/C/D.
2. reference substitution עם donor strata.
3. intermediate-layer causal patching.
4. pairwise conditional/counterfactual composition matrix.
5. cross-context stability + early fingerprint prediction.

כל אחד קיבל מראש competing explanations, predictions ו־falsifiers.

### למה הם לא רצו בפועל

לא נמצאה גישה עיוורת למשקלים:

- בעץ Git אין weight-like checkpoint blobs.
- run `37694378705` מכיל artifact עיוור יחיד בלבד: `four-component-blind-dataset-47`.
- לא נמצא manifest עיוור שממפה opaque checkpoint handles אל V1–V7.

כדי לבצע activation patching היה צריך לפתוח קוד/שמות שמפרים את תנאי המחקר. לכן הניסויים נשארו preregistered אך **NOT EXECUTED**.

## חמשת ה־artifact-only benchmarks שכן רצו

### F1. Direction-vs-magnitude residual

בודק האם mode של A (`zero/negate/shuffle`) מסביר שינוי ב־B מעבר ל־`mean_effect_norm`.

### F2. Learned-reference specificity

בודק ישירות learned vs shuffle/random reference margins ל־B ול־C, בלי composite score.

### F3. Mechanism-parameter coupling and lag

בודק האם `gate`, `path_up_norm`, `path_down_norm` קשורים ל־A effect באותה שכבה, והאם הם מנבאים effect בשכבה הבאה.

### F4. C-vs-D challenge asymmetry

בודק את primitive difference:

`B_under_D_accuracy - B_under_C_accuracy`

בכל שכבה/checkpoint.

### F5. Early fingerprint stability

בודק האם geometry של pairwise distances בין V2–V7 בשלב מוקדם מנבא את geometry ב־120M, בשלושה subspaces נפרדים: intervention, reference, depth/geometry/accessibility.

---

# 3. תוצאות שאישרו את ההשערות

## 3.1 D אכן נראה challenge חזק ועקבי יותר מ־C

זהו הממצא החזק ביותר במחקר ההמשך.

על 294 רשומות בכיסוי 25–120M:

- mean `D-C = -0.04132`
- 95% checkpoint-cluster bootstrap CI: `[-0.05140, -0.03212]`
- negative: `198/294 = 67.35%`
- zero: `96/294 = 32.65%`
- positive: `0/294`

כלומר **לא נמצאה אפילו רשומה אחת שבה B תחת D היה טוב יותר מאשר תחת C**.

כל שבעת הווריאנטים בעלי mean שלילי:

| Variant | mean D-C | fraction negative |
|---|---:|---:|
| V1 | -0.0207 | 0.405 |
| V2 | -0.0397 | 0.714 |
| V3 | -0.0542 | 0.762 |
| V4 | -0.0520 | 0.690 |
| V5 | -0.0357 | 0.786 |
| V6 | -0.0379 | 0.690 |
| V7 | -0.0489 | 0.667 |

Leave-one-variant-out לא שובר את התוצאה: הממוצע נשאר בין `-0.0448` ל־`-0.0392` בכל השמטה.

### פירוש

ההשערה ש־D הוא condition מאתגר יותר מתחזקת משמעותית. אבל זה עדיין **לא מוכיח ש־D “גורם” להפרעה**; הנתון משווה conditions, לא intervention ישיר על D.

## 3.2 ההפרדה בין effect strength לבין learned-value specificity התחזקה

F2 מראה שה־learned reference יכול להיות עקבי מאוד בכיוון שונה בין וריאנטים, גם בלי שינוי מקביל בעוצמת A.

ב־C:

- pooled mean כמעט אפס: `+0.0000045`
- CI: `[-0.00309, +0.00281]`

אבל לפי וריאנט:

- V5: mean `-0.01573`, שלילי `42/42`
- V3: mean `-0.00386`, שלילי `37/42`
- V4: mean `+0.00729`, חיובי `40/42`
- V6: mean `+0.01149`, חיובי `41/42`
- V2: כמעט אפס
- V7: כמעט ניטרלי/מעורב

זה מאשר חזק את העיקרון מן המחקר הראשון: **כמה חזק A משפיע וכמה הערך המסוים של A חשוב הם שני דברים שונים.**

## 3.3 משפחת robustness של V6–V7 נשארת נראית

בבדיקת post-hoc שמפרקת את F1 לפי וריאנט:

- V6: mean `negate-zero B accuracy = 0.000`
- V7: mean `negate-zero B accuracy = 0.000`

לעומת:

- V2: `-0.0714`
- V3: `-0.0992`
- V4: `-0.0833`
- V5: `-0.0556`

לכן ההבחנה בין V6/V7 לבין V2–V5 מבחינת B robustness תחת A נותרת שימושית.

---

# 4. תוצאות שהחלישו את ההשערות

## 4.1 “A רגיש לסימן” היה סיפור חזק מדי

F1 הוא כנראה התיקון החשוב ביותר לפרשנות הראשונה.

מודל magnitude-only בתוך checkpoint-layer נותן:

- `R² = 0.2484`

לאחר הוספת mode indicators:

- `R² = 0.2690`
- SSE משתפר רק ב־`2.74%`
- leave-one-group-out SSE משתפר רק ב־`2.08%`

המקדמים לאחר תיקון ל־effect norm:

- effect norm: `-0.2284`, CI `[-0.3054, -0.1455]`
- negate residual: `+0.0503`, CI `[+0.0145, +0.0834]`
- shuffle residual: `-0.00123`, CI `[-0.00405, +0.00126]`

### המשמעות

הנזק הגולמי של `negate` ל־B אינו נראה בעיקר כמו “הסימן הלא נכון”. הוא נראה בעיקר כמו **perturbation גדול יותר**.

אפילו יותר מעניין: אחרי תיקון לגודל, negate residual חיובי. כלומר עבור perturbation באותה עוצמה צפויה, `negate` מזיק **פחות** מכפי שמודל magnitude-only היה מנבא.

זה לא אומר שכיוון לא משנה בכלל. הוא מוסיף מעט מידע. אבל זה מחליש משמעותית טענה של “sign-sensitive A” כמנגנון כללי.

## 4.2 “A הוא reference אוניברסלי” נחלש

עבור B יש pooled learned advantage קטן וחיובי:

- mean `+0.00123`
- CI `[+0.00039, +0.00202]`

אבל mean absolute learned advantage הוא `0.00261`, קטן מ־mean absolute shuffle-random difference `0.00375`.

עבור C אין pooled advantage בכלל.

מסקנה: **specific learned reference הוא property ארכיטקטוני/וריאנטי, לא universal role של A.**

לכן הביטוי “A = reference” צריך לרדת בביטחון. ניסוח זהיר יותר הוא: A הוא dynamic intervention-sensitive axis שלחלק מהווריאנטים יש עליו learned-value semantics חזקים, ולחלקם polarity הפוכה או כמעט אפסית.

---

# 5. תוצאות שהפריכו השערות

## 5.1 simple gate/path -> next-layer-effect mechanism הופרך

F3 בדק האם הפרמטרים העיוורים `gate`, `path_up_norm`, `path_down_norm` בשכבה L מנבאים A effect בשכבה L+1.

### same-layer

- gate centered Spearman: `-0.066`
- path_down: `+0.142`
- path_up: `+0.234`
- regression `R² = 0.0644`

יש קשר חלש, בעיקר ל־path_up.

### lagged

- lag regression `R² = 0.0186`
- permutation null mean `R² = 0.0352`
- permutation `p = 0.628`

כלומר המודל עם פרמטרי שכבה L **לא טוב יותר מן הסדר האקראי** בניבוי effect בשכבה L+1.

### מסקנה

הסיפור ההתפתחותי “פרמטרי המנגנון מתחזקים קודם, ואז A effect עוקב בשכבה הבאה” **אינו נתמך** במדדים האלה.

אפשרויות חלופיות:

- ה־norms אינם הפרמטרים הפונקציונליים החשובים.
- הכיוון/אינטראקציה בין matrices חשובים יותר מה־norm.
- ההשפעה מבוזרת ולא מקומית לשכבה.
- A effect נוצר כתוצאה של dynamics שאינה ניתנת להסבר בפרמטר scalar יחיד לכל שכבה.

## 5.2 “fingerprint ארכיטקטוני אחד מתקבע מוקדם ונשמר” הופרך בצורתו הפשוטה

F5 מראה שלושה clocks שונים.

### intervention fingerprint

25M -> 120M:

- Spearman `0.5821`, permutation `p=0.0147`
- Pearson `0.6292`, `p=0.0045`

אבל 50M -> 120M:

- Spearman `0.1857`, `p=0.265`
- Pearson `0.2416`, `p=0.250`

כלומר 25M מנבא את 120M **טוב יותר מ־50M**.

### reference fingerprint

- 25M -> 120M: weak/non-significant
- 50M -> 120M: Spearman `0.425`, `p≈0.090`

ובבדיקת post-hoc על כל הזוגות:

- 50–75: `0.864`
- 50–100: `0.789`
- 50–110: `0.839`
- 75–100: `0.807`
- 75–110: `0.871`
- 100–110: `0.846`

ואז הדמיון ל־120M יורד.

### depth/geometry/accessibility

- 25M -> 120M: Spearman `-0.193`, לא מובהק
- 50M -> 120M: Spearman `0.525`, `p=0.0482`

אבל הזוגות האחרים לא יוצרים תמונה יציבה.

### מסקנה

אין “fingerprint אחד”. יש subspaces שמתפתחים בקצבים שונים.

---

# 6. האם מפת A/B/C/D השתנתה

## A — מפה מעודכנת

### מה נשאר

- A הוא רכיב דינמי עם effect measurable.
- strength, direction consistency ו־specific learned value הם אכן dimensions נפרדים.
- וריאנטים משתמשים ב־A בדרכים שונות מאוד.

### מה השתנה

- confidence ב־“reference אוניברסלי” ירד.
- confidence ב־“sign sensitivity” ירד.
- magnitude sensitivity עלתה בחשיבות.

### Inferred role מעודכן

**Dynamic modulation/control axis עם learned-value semantics שתלויות בארכיטקטורה**, ולא reference אוניברסלי.

### Confidence

- high: strength/value-specificity הם נפרדים.
- medium: A הוא dynamic control/modulation axis.
- low-medium: A הוא reference במובן פונקציונלי כללי.

## B — מפה מעודכנת

אין benchmark חדש שמבצע intervention ישיר על B, ולכן לא קיבלנו causal confirmation של “anchor”.

מה שכן:

- robustness של B תחת A ממשיך להיות גבוה בחלק מהווריאנטים.
- D פוגע ביכולת לזהות B יותר מ־C באופן כמעט חד־צדדי.

**מסקנה:** B עדיין מועמד טוב ל־stable anchor, אך זה נשאר inference תצפיתי עד direct B patching.

## C — מפה מעודכנת

הקשר היחסי ל־B נשאר plausible, אבל learned-A influence על C הוא architecture-specific ומחליף סימן בין וריאנטים.

מסקנה: **C אינו פשוט output שנגזר מ־A**. אם הוא relational, היחס כנראה עובר דרך מנגנון נוסף שתלוי בארכיטקטורה.

Confidence יורד מעט מ־medium ל־medium-low לגבי תפקיד ספציפי, ונשאר medium לגבי “relation-like behavior” במדדים הקיימים.

## D — מפה מעודכנת

D התחזק משמעותית כ־“challenge dimension”.

הנתון החזק הוא לא רק mean נמוך יותר, אלא **אפס counterexamples חיוביים ב־294 רשומות**.

עם זאת, אין direct D intervention. לכן:

- confidence ש־D הוא condition קשה/מתחרה: medium-high.
- confidence ש־D עצמו סיבתית entangled: עדיין low-medium.

---

# 7. האם מפת המשפחות V1–V7 השתנתה

כן. המסקנה החשובה היא שאין partition יחיד. יש כמה axes שחוצים זה את זה.

## Axis 1: robustness של B להתערבות A

- robust: V6, V7
- sensitive: V2, V3, V4, V5
- V1: structurally distinct / A-null במדדים האלה

## Axis 2: learned-A specificity עבור C

- positive: V4, V6
- negative: V3, V5
- near-neutral/mixed: V2, V7
- V1: unavailable

## Axis 3: relation בין A vector norm ל־A effect לאורך האימון — post-hoc

Spearman vector norm vs final A effect:

- V2: `-0.886`
- V3: `-0.943`
- V5: `-0.829`
- V4: `+0.657`
- V6: `+0.943`
- V7: `+0.943`

אבל pooled within-variant-centered correlation קרוב לאפס (`-0.110`).

זה אומר שאין law אוניברסלי של “vector גדול -> effect גדול”. יש לפחות שתי התנהגויות מנגנון הפוכות בין וריאנטים.

## Fingerprints מעודכנים

### V1
עדיין structurally distinct. אין learned-A fields, ואין measured A effect. D challenge קיים אבל חלש יותר בממוצע.

### V2
A-sensitive; רוב raw negate harm מוסבר על ידי norm; learned-C כמעט ניטרלי; vector norm וה־effect נעים הפוך לאורך האימון.

### V3
A-sensitive ביותר בקבוצת V2–V4; learned-C שלילי עקבי; vector norm/effect strongly inverse.

### V4
A-sensitive אך learned-C positive כמעט תמיד; לכן הוא דוגמה טובה לכך ש־intervention sensitivity ו־reference polarity הם axes נפרדים.

### V5
ה־singleton הכי חריג במחקר ההמשך. learned-C שלילי 42/42; אחרי norm control, negate residual דווקא שלילי, בניגוד לשאר רוב הווריאנטים; vector norm/effect inverse. זה מחזק השערה למנגנון שונה.

### V6
B robust ל־negate, learned-C positive כמעט תמיד, vector norm/effect positive. candidate למשפחה שבה A signal מתחזק באופן “aligned” יותר.

### V7
B robust ל־negate, learned-C כמעט ניטרלי, vector norm/effect positive. דומה ל־V6 ב־robustness אך לא ב־reference specificity.

### מסקנה על משפחות

החלוקה הטובה יותר אינה tree יחיד אלא **factorial map**:

- robustness axis
- learned-reference polarity axis
- norm/effect coupling axis

זה מסביר מדוע clustering אחד היה לא יציב במחקר הראשון.

---

# 8. אילו יחסים נראים סיבתיים יותר ואילו רק גאומטריים

## יחס A -> B

יש intervention data על A, ולכן זה היחס היחיד עם causal leverage מסוים ב־artifact.

אבל F1 מראה שהפרשנות הסיבתית צריכה להיות זהירה:

- effect norm הוא driver חזק.
- mode מוסיף מעט.
- shuffle כמעט לא מוסיף מעבר ל־norm.

כלומר יש evidence ש־A perturbation משפיע על B, אבל **לא evidence חזק שהסימן או הזהות הספציפית הם הסיבה המרכזית**.

## B <-> C

עדיין בעיקר geometric/conditional evidence. אין direct B/C intervention חדש.

## B <-> D

F4 נותן conditional asymmetry חזקה מאוד, אך לא intervention. לכן היחס נראה פונקציונלית חשוב, אבל אינו causal proof.

## A <-> C

learned-reference polarity לפי וריאנט היא ראיה חזקה לכך שהיחס אינו trivial, אבל שוב אין intervention חדש על C.

---

# 9. אילו שכבות נראות פונקציונליות ולא רק decodable

המחקר הזה **לא יכול להוכיח פונקציונליות של שכבות ביניים** בלי activation patching.

מה שכן אפשר לומר:

## 9.1 simple parameter-lag mechanism לא עובד

F3 הראה שה־scalar layer parameters אינם מספקים הסבר פונקציונלי לסדר העומק.

## 9.2 D-vs-C asymmetry היא layer-dependent

הפער D-C לפי שכבה:

| Layer | mean D-C | negative fraction |
|---|---:|---:|
| layer_1 | -0.0220 | 0.571 |
| layer_2 | -0.0677 | 0.810 |
| layer_3 | -0.0134 | 0.452 |
| layer_4 | -0.0467 | 0.810 |
| layer_5 | -0.0437 | 0.667 |
| layer_6 | -0.0538 | 0.714 |
| final | -0.0419 | 0.690 |

זה מעניין במיוחד כי layer 2 שוב נראה כנקודת לחץ, אבל layer 3 כמעט “מתקן” את ההפרש לפני שהוא חוזר בשכבות 4–6.

זה תואם אפשרות של processing chain יותר מורכבת מ־compression יחיד:

**stress/compression -> repair/reorganization -> re-expression**.

אבל ללא patching אי אפשר לקרוא לזה causal chain.

---

# 10. התפתחות לאורך האימון

ה־follow-up מצביע על לפחות שני clocks שונים.

## Clock A: intervention fingerprint

- 25M כבר דומה ל־120M.
- 50M מתרחק משמעותית.
- 75–110M מתקרבים שוב בחלק מההשוואות.
- 100–110M מאוד דומים (`rho=0.861`).

זה מעלה אפשרות של **early attractor -> transient reorganization -> return/reconstruction**.

אני לא מכנה זאת hysteresis מוכח, כי יש רק 6 variants במטריצה ו־15 pairwise distances, אבל הדפוס הלא־מונוטוני ברור מספיק כדי לשלול “התייצבות חלקה”.

## Clock B: learned-reference fingerprint

reference geometry דווקא יציבה מאוד מ־50M עד 110M, ואז הדמיון ל־120M יורד.

זה מרמז שה־specific learned value יכול להתארגן בשלב אחר מה־intervention sensitivity.

## Clock C: depth/geometry/accessibility

המסלול הזה הרבה פחות יציב. אין family geometry אחת שחוזרת עקבית בין token counts.

### מסקנה התפתחותית חדשה

במקום:

> קודם נוצר מנגנון, אחר כך הוא מקבל משמעות, ואז מתחזק

הנתונים מתאימים יותר ל:

> כמה תתי־מערכות מתארגנות במקביל, עם reorganization לא־מונוטוני. intervention sensitivity יכולה להראות fingerprint מוקדם, learned-reference specificity מתייצבת באמצע, ו־depth/accessibility ממשיכים להשתנות בנפרד.

---

# 11. מה ממש הפתיע

## 11.1 negate residual הפך חיובי אחרי תיקון לעוצמה

זה היה counterexample ישיר לסיפור שהסימן עצמו הוא הבעיה. בני אדם, כידוע, אוהבים לתת שם דרמטי ל־`negate` ואז להניח שהשם הוא המנגנון. המספרים לא חייבים לשתף פעולה.

## 11.2 אפס מקרים של D טוב מ־C

294 רשומות, 0 positive. זה הרבה יותר חד ממה שציפיתי מהמחקר הראשון.

## 11.3 25M מנבא intervention geometry ב־120M טוב יותר מ־50M

זה הדפוס ההתפתחותי המוזר ביותר בשלב השני.

## 11.4 gate/path norms כמעט לא מנבאים next-layer effect

הם נראו מועמדים טבעיים להסבר. הם נכשלו.

---

# 12. דפוסים שנראו אמיתיים ואז נשברו

1. **“negate hurts because sign matters”** -> נשבר חלקית. magnitude מסביר את רוב האפקט.
2. **“A is a universal learned reference”** -> נשבר. reference specificity מחליפה polarity לפי וריאנט.
3. **“layer parameters lead the downstream A effect”** -> הופרך ב־lag permutation.
4. **“variant fingerprint stabilizes monotonically”** -> נשבר. 25->120 חזק יותר מ־50->120 ב־intervention subspace.
5. **“one natural clustering of V1–V7”** -> נחלש. לפחות שלושה axes חוצים זה את זה.

---

# 13. השערות חדשות

## H1. A מכיל לפחות שני ערוצים פונקציונליים

אחד קשור ל־perturbation magnitude, והשני ל־mode/identity. הערוץ הראשון דומיננטי ברוב הווריאנטים.

**מבחן מכריע:** same-norm angular rotations ו־same-norm opposite direction על checkpoint weights.

## H2. learned-reference polarity הוא property ארכיטקטוני עצמאי מ־A robustness

V4 ו־V6 reference-positive למרות robustness שונה; V3/V5 reference-negative למרות intervention profiles שונים.

**מבחן מכריע:** donor substitution עם matched norm ו־matched component equality.

## H3. V5 משתמש במנגנון A שונה איכותית

V5 הוא היחיד שבו norm-controlled negate residual שלילי, יחד עם learned-C שלילי 42/42 ו־inverse norm/effect coupling.

**מבחן מכריע:** layer-wise same-norm A patching ב־V5 מול V4/V6.

## H4. D עובר שלב של “repair” סביב layer 3

D disadvantage חזק ב־layer 2, כמעט נעלם ב־layer 3, וחוזר ב־4–6.

**מבחן מכריע:** direct D intervention לפני/אחרי layer 2 ו־3, עם activation patching.

## H5. intervention fingerprint עובר transient reorganization סביב 50M

25M ו־120M דומים יותר מאשר 50M ו־120M.

**מבחן מכריע:** checkpoints צפופים בין 25–75M, ללא אימון חדש אם קיימים snapshots, ומדידת אותו primitive intervention vector.

## H6. reference specificity מתארגנת ב־mid-training ועוברת reconfiguration סמוך ל־120M

50–110M מראים pairwise geometry יציבה מאוד, ואז ירידה מול 120M.

**מבחן מכריע:** 105/115/117.5/120M אם קיימים snapshots, או checkpoints הקיימים הצפופים ביותר באזור זה.

## H7. scalar parameter norms אינם המשתנים המכניסטיים הנכונים

הכיוון, alignment בין matrices, singular structure או interaction terms עשויים להיות חשובים יותר.

**מבחן מכריע:** blind-safe matrix-level primitives, ללא שמות סמנטיים.

---

# 14. ניסויים מכריעים שעדיין נדרשים

הניסויים בעלי הערך הגבוה ביותר נשארים אלה שנרשמו מראש לפני התוצאות:

1. **same-norm intervention matrix על A/B/C/D** עם random orthogonal / opposite / interpolation / extrapolation.
2. **direct B intervention** כדי לבדוק אם B באמת anchor ולא רק decodable feature.
3. **C/D matched interventions** כדי להפוך את F4 מ־conditional asymmetry ל־causal comparison.
4. **reference substitution** לפי same-B/different-C וכדומה, דרך API עיוור בלבד.
5. **intermediate-layer activation patching** כדי לבדוק אם layer 2/3/4 הם causal stages.
6. **cross-context sequence tests** למדידת timescales אמיתיים של A/B/C/D.

לשם כך נדרש blind runner קטן שמחזיר רק V1–V7, A–D ו־opaque checkpoint handles. מפרט כזה מצורף לחבילת המחקר.

---

# 15. שאלות שנשארו פתוחות

1. האם B הוא באמת causal anchor, או רק feature שקל מאוד לפענח?
2. האם C נבנה ביחס ל־B או ששני המדדים חולקים geometry משותפת?
3. האם D הוא רכיב מתחרה, או פשוט condition שמייצר samples קשים יותר?
4. מדוע layer 3 מצמצם את D-vs-C gap ואז הפער חוזר?
5. מה בדיוק קורה סביב 50M ב־intervention fingerprint?
6. מדוע reference fingerprint יציב ב־50–110M ואז משתנה ב־120M?
7. מהו המשתנה המכניסטי האמיתי של A אם gate/path norms לא מנבאים את ההשפעה?
8. האם V5 הוא מנגנון נפרד או קצה קיצוני של continuum?
9. האם V6/V7 robustness נובע מ־independence אמיתי או פשוט saturation של B accuracy?
10. האם הדפוסים נשמרים תחת prompt/context banks חדשים, שאין ב־artifact הנוכחי?

---

# 16. 130M החלקי

130M לא שימש לשום ranking מלא או inference על שבעת הווריאנטים.

הכיסוי החלקי הוא V2, V3, V5, V6, V7 בלבד.

בדיקה descriptive מול 120M מראה:

- סימן learned-C specificity נשמר בכל 5/5 הווריאנטים הזמינים.
- סימן D-C נשמר בכל 5/5.

זה מספק sanity check חלקי בלבד, לא הוכחה להמשך מגמה.

---

# 17. מגבלות וביקורת עצמית

## 17.1 המגבלה המרכזית

לא בוצעו interventions חדשים על המשקלים. לכן F1–F5 הם **independent falsification analyses על primitive blind measurements קיימים**, לא generation של assay חדש על המודל עצמו.

## 17.2 F1 הוא adjustment סטטיסטי, לא true magnitude matching

ה־regression מתקן ל־`mean_effect_norm`; הוא אינו מחליף perturbations שנבנו מראש להיות equal-norm.

## 17.3 F2 משתמש ב־reference modes שכבר קיימים ב־artifact

החדש כאן הוא ה־test design והפירוק primitive, לא יצירת donor bank חדש.

## 17.4 F5 קטן

V2–V7 נותנים 6 variants בלבד, כלומר 15 pairwise distances. permutation test עוזר, אבל אי אפשר להעמיס על זה ביטחון של dataset גדול.

## 17.5 post-hoc analyses מסומנים בנפרד

כל מה שמעבר ל־F1–F5, כולל per-variant residual decomposition, כל-pairs developmental matrix, vector-norm/effect coupling ו־130M descriptive, מסומן exploratory ולא משנה את predictions שנקבעו מראש.

---

# 18. מסקנה סופית

לאחר שלב ההמשך, המפה העיוורת שלי היא:

- **A:** dynamic modulation/control axis; magnitude חשוב מאוד; learned-value semantics קיימים אך architecture-dependent. פחות בטוח כ־universal reference.
- **B:** עדיין המועמד החזק ביותר לעוגן יציב, אך direct causal test עדיין חסר.
- **C:** relation-like behavior נשאר plausible, אבל הקשר ל־A תלוי וריאנט ומחליף polarity.
- **D:** challenge dimension חזק ועקבי יותר ממה שהוערך קודם; direct causality עדיין לא הוכחה.

המפה של V1–V7 השתנתה ממספר “משפחות” פשוטות למבנה רב־צירי:

- A-intervention robustness
- learned-reference polarity
- norm/effect coupling

וההתפתחות לאורך האימון נראית כעת **לא־מונוטונית ורב־שלבית**: intervention fingerprint, reference specificity ו־depth/accessibility אינם מתארגנים באותו קצב.

הדבר החשוב ביותר שהמחקר השני הוסיף אינו “עוד אישור”. הוא הסיר שלושה הסברים פשוטים מדי: sign-only sensitivity, universal reference, ו־simple layer-parameter lead-lag mechanism.

**המחקר הסתיים ללא unblinding.**
