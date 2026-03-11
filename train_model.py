"""
train_model.py  ·  AI Health Guardian
Perfectly tailored for:  dataset.csv  (itachi9604 Kaggle format)
  - 4,920 rows × 18 columns
  - Column 0  : 'Disease'   (target — 41 classes, may have trailing spaces)
  - Columns 1-17: 'Symptom_1' ... 'Symptom_17'  (text symptom names, leading spaces, NaN)
  - 131 unique symptoms after cleaning

Run once:  python train_model.py
"""

import pandas as pd
import numpy as np
import pickle, json, os, time, sys
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# ── Terminal colours ──────────────────────────────────────────────────────────
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    GREEN  = Fore.GREEN  + Style.BRIGHT
    CYAN   = Fore.CYAN   + Style.BRIGHT
    YELLOW = Fore.YELLOW + Style.BRIGHT
    BLUE   = Fore.BLUE   + Style.BRIGHT
    MAGENTA= Fore.MAGENTA+ Style.BRIGHT
    WHITE  = Fore.WHITE  + Style.BRIGHT
    DIM    = Style.DIM
    RESET  = Style.RESET_ALL
except ImportError:
    GREEN=CYAN=YELLOW=BLUE=MAGENTA=WHITE=DIM=RESET=""

def banner():
    print(f"""
{CYAN}╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   {WHITE}🏥  AI Health Guardian  ·  Model Training Pipeline{CYAN}            ║
║   {DIM}SDG 3 : Good Health and Well-Being{CYAN}                           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝{RESET}
""")

def section(title):
    pad = 60
    print(f"\n{BLUE}┌─{'─'*len(title)}─┐\n│ {WHITE}{title}{BLUE} │\n└─{'─'*len(title)}─┘{RESET}")

def ok(msg):   print(f"  {GREEN}✔  {RESET}{msg}")
def warn(msg): print(f"  {YELLOW}⚠  {RESET}{msg}")
def info(msg): print(f"  {CYAN}ℹ  {RESET}{msg}")

def progress_bar(label, total=25, delay=0.012):
    sys.stdout.write(f"  {CYAN}{label}  [{RESET}")
    for _ in range(total):
        time.sleep(delay)
        sys.stdout.write(f"{GREEN}█{RESET}")
        sys.stdout.flush()
    sys.stdout.write(f"{CYAN}]  {GREEN}Done!{RESET}\n")

def accuracy_bar(name, acc, best):
    filled = int(acc * 30)
    color  = GREEN if acc == best else CYAN if acc > 0.85 else YELLOW
    bar    = f"{color}{'█'*filled}{DIM}{'░'*(30-filled)}{RESET}"
    star   = f"  {YELLOW}★ BEST{RESET}" if acc == best else ""
    print(f"  {WHITE}{name:<25}{RESET} {bar}  {color}{acc*100:5.2f}%{RESET}{star}")

# ═══════════════════════════════════════════════════════════════════════════════
banner()
os.makedirs("model", exist_ok=True)
os.makedirs("data",  exist_ok=True)

# ── STEP 1: Load ──────────────────────────────────────────────────────────────
section("STEP 1 — Loading Dataset")

DATASET_PATHS = [
    "data/dataset.csv", "data/Training.csv",
    "dataset.csv", "Training.csv",
]
df = None
for path in DATASET_PATHS:
    if os.path.exists(path):
        df = pd.read_csv(path, dtype=str)          # load everything as string
        ok(f"Loaded: {WHITE}{path}{RESET}  →  {df.shape[0]:,} rows × {df.shape[1]} columns")
        break

if df is None:
    print(f"  {YELLOW}⚠  No CSV found. Place dataset.csv inside the data/ folder.{RESET}")
    sys.exit(1)

# ── STEP 2: Inspect ───────────────────────────────────────────────────────────
section("STEP 2 — Inspecting Dataset")

TARGET_COL   = "Disease"
SYMPTOM_COLS = [c for c in df.columns if c != TARGET_COL]

info(f"Target column  : {WHITE}{TARGET_COL}{RESET}")
info(f"Symptom columns: {WHITE}{len(SYMPTOM_COLS)}{RESET}  ({SYMPTOM_COLS[0]} … {SYMPTOM_COLS[-1]})")
info(f"Diseases found : {WHITE}{df[TARGET_COL].nunique()}{RESET}")

# ── STEP 3: Clean & build binary matrix ───────────────────────────────────────
section("STEP 3 — Cleaning Data & Building Binary Feature Matrix")

def clean(val):
    """Strip spaces, lowercase, normalise underscores."""
    if pd.isna(val) or str(val).strip() in ("", "nan", "None"):
        return ""
    return str(val).strip().lower().replace(" ", "_")

# Collect every unique (cleaned) symptom name
info("Collecting all unique symptoms across 17 columns...")
all_symptoms_set = set()
for col in SYMPTOM_COLS:
    vals = df[col].apply(clean)
    all_symptoms_set.update(v for v in vals if v)

ALL_SYMPTOMS = sorted(all_symptoms_set)
ok(f"Unique symptoms found: {WHITE}{len(ALL_SYMPTOMS)}{RESET}")

# Build binary matrix — one column per symptom, 1 if present in any Symptom_X
info("Building one-hot symptom matrix (this may take a moment)...")
rows_data = []
for _, row in df.iterrows():
    present = {clean(row[col]) for col in SYMPTOM_COLS}
    present.discard("")
    rows_data.append({s: (1 if s in present else 0) for s in ALL_SYMPTOMS})

X_df = pd.DataFrame(rows_data, dtype=np.int8)
ok(f"Binary matrix: {WHITE}{X_df.shape[0]:,} rows × {X_df.shape[1]} symptom features{RESET}")

# Clean target labels (strip trailing/leading spaces)
y_raw = df[TARGET_COL].apply(lambda v: str(v).strip())

ok(f"Sample label check: {list(y_raw.unique()[:5])}")

# ── STEP 4: Encode labels ─────────────────────────────────────────────────────
section("STEP 4 — Label Encoding")

le    = LabelEncoder()
y_enc = le.fit_transform(y_raw)

ok(f"Total classes   : {WHITE}{len(le.classes_)}{RESET}")
print(f"\n  {CYAN}All 41 diseases:{RESET}")
for i, d in enumerate(le.classes_):
    end = "\n" if (i+1) % 3 == 0 or i == len(le.classes_)-1 else ""
    print(f"  {DIM}{i+1:>2}.{RESET} {d:<40}", end=end)

# ── STEP 5: Train/Test Split ──────────────────────────────────────────────────
section("STEP 5 — Train / Test Split  (80% / 20%)")

X = X_df.values
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
ok(f"Training samples : {WHITE}{len(X_train):,}{RESET}")
ok(f"Test samples     : {WHITE}{len(X_test):,}{RESET}")

# ── STEP 6: Train all models ──────────────────────────────────────────────────
section("STEP 6 — Training 4 ML Models")

MODELS = {
    "Random Forest":     RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Decision Tree":     DecisionTreeClassifier(random_state=42),
    "Naive Bayes":       GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results, best_model, best_acc, best_name = {}, None, 0.0, ""

for name, clf in MODELS.items():
    progress_bar(f"Training {name:<22}")
    t0 = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0
    acc = accuracy_score(y_test, clf.predict(X_test))
    cv  = cross_val_score(clf, X, y_enc, cv=5, scoring="accuracy")
    results[name] = {"accuracy": acc, "cv_mean": cv.mean(), "cv_std": cv.std(), "time": elapsed}
    if acc > best_acc:
        best_acc, best_model, best_name = acc, clf, name

# ── STEP 7: Results ───────────────────────────────────────────────────────────
section("STEP 7 — Model Comparison")

print(f"\n  {WHITE}{'Model':<25} {'Accuracy':>9} {'CV Mean':>9} {'CV Std':>8} {'Time':>7}{RESET}")
print(f"  {'─'*62}")
for name, r in results.items():
    star  = f"  {YELLOW}★{RESET}" if name == best_name else ""
    color = GREEN if name == best_name else CYAN
    print(f"  {color}{name:<25}{RESET}"
          f" {r['accuracy']*100:8.2f}%"
          f"  {r['cv_mean']*100:8.2f}%"
          f"  ±{r['cv_std']*100:4.2f}%"
          f"  {r['time']:5.1f}s{star}")

print(f"\n  {MAGENTA}Visual Accuracy Comparison:{RESET}")
for name, r in results.items():
    accuracy_bar(name, r["accuracy"], best_acc)

print(f"\n  {GREEN}🏆 Best: {WHITE}{best_name}{RESET}  →  {GREEN}{best_acc*100:.2f}% accuracy{RESET}")

# ── STEP 8: Classification report ────────────────────────────────────────────
section("STEP 8 — Detailed Classification Report")
y_pred_best = best_model.predict(X_test)
print(f"\n{DIM}{classification_report(y_test, y_pred_best, target_names=le.classes_, digits=3)}{RESET}")

# ── STEP 9: Feature importance ────────────────────────────────────────────────
if hasattr(best_model, "feature_importances_"):
    section("STEP 9 — Top 20 Most Predictive Symptoms")
    imp     = best_model.feature_importances_
    top_idx = np.argsort(imp)[::-1][:20]
    max_imp = imp[top_idx[0]]
    for rank, idx in enumerate(top_idx, 1):
        bl  = int(imp[idx] / max_imp * 28)
        bar = f"{GREEN}{'█'*bl}{DIM}{'░'*(28-bl)}{RESET}"
        print(f"  {WHITE}{rank:>2}.{RESET} {ALL_SYMPTOMS[idx]:<38} {bar}  {DIM}{imp[idx]:.5f}{RESET}")

# ── STEP 10: Save artifacts ───────────────────────────────────────────────────
section("STEP 10 — Saving All Artifacts")
progress_bar("Saving model + encoder + metadata", delay=0.02)

with open("model/disease_model.pkl",    "wb") as f: pickle.dump(best_model, f)
with open("model/label_encoder.pkl",    "wb") as f: pickle.dump(le, f)
with open("model/symptom_columns.json", "w")  as f: json.dump(ALL_SYMPTOMS, f, indent=2)

model_meta = {
    "best_model":     best_name,
    "accuracy":       round(best_acc, 4),
    "n_diseases":     int(len(le.classes_)),
    "n_symptoms":     int(len(ALL_SYMPTOMS)),
    "dataset_format": "text_symptom_columns",
    "diseases":       list(le.classes_),
    "all_symptoms":   ALL_SYMPTOMS,
    "results": {
        k: {kk: round(vv, 4) for kk, vv in v.items()}
        for k, v in results.items()
    }
}
with open("model/model_info.json", "w") as f: json.dump(model_meta, f, indent=2)

ok(f"model/disease_model.pkl        ({os.path.getsize('model/disease_model.pkl')//1024} KB)")
ok("model/label_encoder.pkl")
ok("model/symptom_columns.json      (131 symptoms)")
ok("model/model_info.json")

# ── STEP 11: Build disease info database ─────────────────────────────────────
section("STEP 11 — Building Disease Info Database (41 diseases)")

DISEASE_INFO = {
    "Fungal infection": {"description": "A fungal infection caused by various fungi affecting skin, nails, or internal organs.", "precautions": ["Keep skin dry and clean", "Avoid sharing personal items", "Wear breathable fabrics", "Use antifungal powder in moist areas"], "treatment": "Antifungal creams or oral antifungal medications prescribed by a doctor.", "diet": "Reduce sugar intake, eat probiotic-rich foods like yogurt, avoid processed foods."},
    "Allergy": {"description": "An immune system reaction to substances (allergens) that are usually harmless.", "precautions": ["Identify and avoid allergens", "Carry antihistamines", "Keep environment dust-free", "Wear a medical alert bracelet"], "treatment": "Antihistamines, corticosteroids, decongestants, immunotherapy for long-term.", "diet": "Anti-inflammatory foods, avoid known food triggers, eat quercetin-rich foods (apples, onions)."},
    "GERD": {"description": "Gastroesophageal reflux disease — chronic acid reflux where stomach acid flows into the esophagus.", "precautions": ["Avoid lying down after meals", "Eat smaller meals", "Avoid spicy and fatty foods", "Elevate head while sleeping"], "treatment": "Antacids, PPIs (omeprazole), H2 blockers, lifestyle changes.", "diet": "Avoid coffee, alcohol, citrus, spicy foods. Eat oatmeal, bananas, ginger, green vegetables."},
    "Chronic cholestasis": {"description": "A liver condition where bile flow is reduced or blocked, causing bile to accumulate.", "precautions": ["Avoid alcohol completely", "Take fat-soluble vitamin supplements", "Regular liver function tests", "Avoid hepatotoxic drugs"], "treatment": "Ursodeoxycholic acid, cholestyramine for itch, treat underlying cause.", "diet": "Low-fat diet, fat-soluble vitamins A/D/E/K supplementation, avoid alcohol."},
    "Drug Reaction": {"description": "An adverse reaction to medication, ranging from mild skin rash to life-threatening anaphylaxis.", "precautions": ["Always inform doctors of all allergies", "Carry a medical ID card", "Never self-medicate", "Report any new symptoms after starting medication"], "treatment": "Stop the offending drug immediately, antihistamines, corticosteroids, epinephrine for anaphylaxis.", "diet": "Stay well-hydrated, avoid alcohol, eat light easily digestible foods during recovery."},
    "Peptic ulcer diseae": {"description": "Sores (ulcers) that develop on the lining of the stomach or upper small intestine.", "precautions": ["Avoid NSAIDs and aspirin", "Quit smoking", "Limit alcohol consumption", "Manage stress effectively"], "treatment": "PPIs, H2 blockers, antibiotics if H. pylori is detected, antacids.", "diet": "Avoid spicy, acidic, fried foods. Eat fiber-rich foods, cabbage juice, honey, probiotic yogurt."},
    "AIDS": {"description": "Advanced stage of HIV infection severely compromising the immune system.", "precautions": ["Strict adherence to ART medications", "Safe sex practices always", "Never share needles or syringes", "Regular CD4 count monitoring"], "treatment": "Antiretroviral therapy (ART) — lifelong treatment; treat opportunistic infections.", "diet": "High protein, high calorie, nutrient-dense foods; food safety is critical due to low immunity."},
    "Diabetes ": {"description": "Chronic metabolic disease where the body cannot properly regulate blood sugar levels.", "precautions": ["Monitor blood glucose daily", "Take medications as prescribed", "Exercise 30 minutes daily", "Avoid refined sugar and white flour"], "treatment": "Type 1: Insulin therapy. Type 2: Oral medications (metformin), insulin, lifestyle changes.", "diet": "Low glycemic index foods, high fiber, lean proteins. Avoid sugary drinks, white rice, refined carbs."},
    "Gastroenteritis": {"description": "Inflammation of the stomach and intestines usually caused by viral or bacterial infection.", "precautions": ["Wash hands before eating and after toilet", "Drink only clean/boiled water", "Avoid contaminated food", "Isolate from others if contagious"], "treatment": "Oral rehydration therapy (ORS), rest, antibiotics only if bacterial cause confirmed.", "diet": "BRAT diet: bananas, rice, applesauce, toast. Clear broths, coconut water, avoid dairy and spicy food."},
    "Bronchial Asthma": {"description": "Chronic inflammation of the airways making breathing difficult with recurring attacks.", "precautions": ["Identify and avoid personal triggers", "Always keep rescue inhaler nearby", "Use air purifiers at home", "Monitor peak flow regularly"], "treatment": "Short-acting bronchodilators for attacks, inhaled corticosteroids for prevention.", "diet": "Omega-3 rich foods (fish, flaxseed), vitamin D, magnesium-rich foods; avoid sulfites and processed foods."},
    "Hypertension ": {"description": "Persistently high blood pressure that damages arteries and increases risk of heart disease.", "precautions": ["Reduce sodium intake to <2g/day", "Exercise 150 min/week", "Manage stress with meditation", "Monitor blood pressure at home"], "treatment": "ACE inhibitors, beta-blockers, diuretics, calcium channel blockers, DASH diet.", "diet": "DASH diet: fruits, vegetables, whole grains, low-fat dairy. Limit salt, alcohol, and saturated fat."},
    "Migraine": {"description": "Recurring severe headache attacks often with nausea, vomiting, and sensitivity to light/sound.", "precautions": ["Keep a migraine diary to identify triggers", "Maintain regular sleep schedule", "Stay well hydrated", "Reduce screen time and bright light exposure"], "treatment": "Triptans (sumatriptan) for acute attacks, preventive medications (topiramate, beta-blockers).", "diet": "Avoid red wine, aged cheese, MSG, caffeine withdrawal. Stay hydrated, regular meal times."},
    "Cervical spondylosis": {"description": "Age-related degeneration of spinal discs and vertebrae in the neck causing pain and stiffness.", "precautions": ["Maintain correct posture", "Use ergonomic workstation setup", "Avoid carrying heavy loads on head/shoulders", "Do gentle neck stretches daily"], "treatment": "Physical therapy, NSAIDs, muscle relaxants, cervical collar, surgery in severe cases.", "diet": "Calcium and vitamin D-rich foods, omega-3 anti-inflammatory foods, turmeric and ginger."},
    "Paralysis (brain hemorrhage)": {"description": "Loss of muscle function in part of the body caused by bleeding within the brain.", "precautions": ["Control blood pressure strictly", "Avoid smoking and heavy alcohol", "Manage diabetes and cholesterol", "Seek emergency care immediately for stroke symptoms"], "treatment": "EMERGENCY surgery, clot removal, blood pressure management, intensive rehabilitation therapy.", "diet": "Low-sodium, heart-healthy diet. Adequate protein for muscle recovery, antioxidant-rich foods."},
    "Jaundice": {"description": "Yellowing of skin and eyes caused by excess bilirubin, indicating liver or bile duct problems.", "precautions": ["Avoid alcohol completely", "Drink only safe water", "Rest adequately", "Avoid fatty and oily foods"], "treatment": "Treat the underlying cause (hepatitis, gallstones, etc.); supportive hydration and nutrition.", "diet": "High carbohydrate, very low fat diet. Fresh fruits, vegetables, sugarcane juice, coconut water."},
    "Malaria": {"description": "Life-threatening parasitic disease spread by Anopheles mosquito bites caused by Plasmodium.", "precautions": ["Sleep under insecticide-treated nets", "Apply DEET repellent on exposed skin", "Take prescribed antimalarials when in endemic areas", "Eliminate standing water around home"], "treatment": "Artemisinin-based combination therapies (ACTs); chloroquine in sensitive regions.", "diet": "High calorie, high protein foods; plenty of fluids and electrolytes; avoid solid foods during high fever."},
    "Chicken pox": {"description": "Highly contagious viral infection causing an itchy blister-like rash all over the body.", "precautions": ["Isolate until all sores crust over", "Avoid scratching to prevent scarring", "Keep nails short and clean", "Get varicella vaccine"], "treatment": "Antihistamines for itch, calamine lotion, acyclovir for severe or at-risk cases.", "diet": "Soft cool foods (yogurt, smoothies), plenty of fluids. Avoid salty, spicy, acidic foods."},
    "Dengue": {"description": "Mosquito-borne viral disease causing high fever, severe headache, joint pain, and rash.", "precautions": ["Use mosquito repellent (DEET) on skin", "Wear full-sleeve clothing", "Use mosquito nets", "Eliminate stagnant water breeding sites"], "treatment": "Supportive care only: rest, hydration, paracetamol for fever. AVOID aspirin and ibuprofen.", "diet": "Papaya leaf extract juice, pomegranate juice, coconut water, ORS, kiwi (for platelet boost)."},
    "Typhoid": {"description": "Serious bacterial infection from Salmonella typhi causing prolonged fever and systemic illness.", "precautions": ["Drink only boiled or bottled water", "Avoid raw street food", "Wash hands thoroughly before eating", "Get typhoid vaccine before travel to endemic areas"], "treatment": "Antibiotics: ciprofloxacin, azithromycin, or ceftriaxone; 7-14 days course; rest and hydration.", "diet": "High calorie soft foods: boiled rice, bananas, boiled potatoes, soups. Avoid raw/spicy/fried foods."},
    "hepatitis A": {"description": "Viral liver infection spread through contaminated food and water, usually self-limiting.", "precautions": ["Wash hands before eating and after toilet", "Drink only safe water", "Avoid raw shellfish", "Get Hepatitis A vaccine"], "treatment": "Supportive care, rest, adequate nutrition, avoid alcohol. No specific antiviral needed.", "diet": "High carbohydrate, very low fat diet. Fresh juices, fruits, vegetables. Absolutely no alcohol."},
    "Hepatitis B": {"description": "Viral liver infection spread through blood, sexual contact, and mother-to-child transmission.", "precautions": ["Get vaccinated (3-dose series)", "Practice safe sex", "Never share needles", "Don't share razors or toothbrushes"], "treatment": "Antiviral medications (tenofovir, entecavir) for chronic cases; regular liver monitoring.", "diet": "Low fat, high protein, avoid alcohol and raw shellfish. Liver-supportive foods: beets, carrots, leafy greens."},
    "Hepatitis C": {"description": "Viral liver infection primarily spread through blood-to-blood contact, often leading to chronic liver disease.", "precautions": ["Never share needles or drug equipment", "Use gloves when handling blood", "Ensure sterile medical/dental procedures", "Regular liver function tests"], "treatment": "Direct-acting antivirals (DAAs) — 8-12 week treatment with >95% cure rate.", "diet": "Liver-friendly diet: avoid alcohol, processed foods, excess sodium. Eat fruits, vegetables, whole grains."},
    "Hepatitis D": {"description": "Liver infection caused by Hepatitis D virus, only occurs in people already infected with Hepatitis B.", "precautions": ["Hepatitis B vaccination also prevents D", "Avoid sharing needles", "Safe sex practices", "Regular liver monitoring if HBV+"], "treatment": "Pegylated interferon-alpha; treat underlying Hepatitis B infection.", "diet": "Same as Hepatitis B: low fat, no alcohol, high antioxidant foods."},
    "Hepatitis E": {"description": "Liver infection from Hepatitis E virus, spread through contaminated water, common in developing regions.", "precautions": ["Drink boiled or purified water only", "Practice strict food hygiene", "Avoid raw shellfish", "Wash hands frequently with soap"], "treatment": "Usually self-limiting in 4-6 weeks; rest and hydration; ribavirin for severe/chronic cases.", "diet": "Low fat, easily digestible foods. Avoid alcohol completely. Stay well hydrated."},
    "Alcoholic hepatitis": {"description": "Liver inflammation caused by excessive alcohol consumption over time.", "precautions": ["Stop drinking alcohol immediately and permanently", "Seek addiction counseling/support", "Nutritional rehabilitation", "Regular liver function monitoring"], "treatment": "Complete alcohol abstinence, corticosteroids, nutritional support, liver transplant in end-stage.", "diet": "High calorie, high protein diet. Vitamins B1 (thiamine), B12, folate supplementation. No alcohol ever."},
    "Tuberculosis": {"description": "Contagious bacterial infection caused by Mycobacterium tuberculosis, primarily affecting lungs.", "precautions": ["Complete the FULL 6-month antibiotic course without missing doses", "Cover mouth with elbow when coughing", "Ensure good room ventilation", "Get tested if exposed to TB patient"], "treatment": "DOTS therapy: 6-9 months of HRZE antibiotics (Isoniazid, Rifampicin, Pyrazinamide, Ethambutol).", "diet": "High calorie, high protein diet. Vitamin D, zinc, iron, B-vitamin supplementation. Avoid alcohol."},
    "Common Cold": {"description": "Mild viral respiratory infection of the upper respiratory tract, usually caused by rhinovirus.", "precautions": ["Wash hands frequently for 20+ seconds", "Avoid touching face", "Avoid close contact with infected people", "Disinfect frequently touched surfaces"], "treatment": "Symptomatic only: rest, hydration, decongestants, throat lozenges, saline nasal spray.", "diet": "Warm soups and broths, herbal teas with honey, citrus fruits, ginger-garlic, hot water with turmeric."},
    "Pneumonia": {"description": "Serious infection inflaming air sacs in lungs, filling them with fluid or pus.", "precautions": ["Get pneumococcal and flu vaccines", "Wash hands frequently", "Don't smoke", "Seek early medical care for respiratory symptoms"], "treatment": "Antibiotics for bacterial, antivirals for viral pneumonia. Hospitalization if oxygen levels drop.", "diet": "Warm fluids, broths, vitamin C rich fruits, easily digestible light foods. Avoid dairy if congested."},
    "Dimorphic hemmorhoids(piles)": {"description": "Swollen and inflamed veins in the rectum and anus causing pain, itching, and bleeding.", "precautions": ["Eat high-fiber diet daily", "Drink 8+ glasses of water", "Avoid straining during bowel movements", "Exercise regularly; avoid prolonged sitting"], "treatment": "Sitz baths, topical creams (hydrocortisone), fiber supplements, sclerotherapy, surgery if severe.", "diet": "High fiber: whole grains, legumes, vegetables, fruits. Drink 2-3L water daily. Avoid spicy food."},
    "Heart attack": {"description": "Emergency: blockage of blood supply to heart muscle causing permanent damage or death.", "precautions": ["CALL EMERGENCY SERVICES IMMEDIATELY", "Chew aspirin 325mg if not allergic", "Keep patient calm and still", "Begin CPR if person is unconscious and not breathing"], "treatment": "EMERGENCY: Thrombolytics, angioplasty/stent, bypass surgery, cardiac medications, ICU care.", "diet": "Post-recovery: Mediterranean diet — olive oil, fish, whole grains, vegetables, nuts. Limit red meat, salt."},
    "Varicose veins": {"description": "Enlarged, twisted, bulging veins (usually in legs) due to weakened vein walls and valves.", "precautions": ["Elevate legs above heart level when resting", "Avoid prolonged standing or sitting", "Wear graduated compression stockings", "Exercise regularly (walking, swimming)"], "treatment": "Compression stockings, sclerotherapy, endovenous laser treatment, surgical stripping.", "diet": "High fiber, low sodium diet. Foods with rutin (buckwheat, asparagus). Maintain healthy weight."},
    "Hypothyroidism": {"description": "Underactive thyroid gland that produces insufficient thyroid hormone, slowing body functions.", "precautions": ["Take levothyroxine at same time every day on empty stomach", "Regular TSH blood tests", "Avoid excess soy and raw cruciferous vegetables", "Inform all doctors about your condition"], "treatment": "Levothyroxine (synthetic T4) hormone replacement — lifelong in most cases.", "diet": "Selenium-rich foods (Brazil nuts, tuna), iodine (seaweed, dairy), zinc. Limit raw kale, broccoli, soy."},
    "Hyperthyroidism": {"description": "Overactive thyroid gland producing excess thyroid hormone, speeding up body functions.", "precautions": ["Regular thyroid function monitoring", "Avoid iodine-rich foods and supplements", "Protect eyes from sun (Graves disease)", "Manage stress and get adequate rest"], "treatment": "Antithyroid drugs (methimazole), radioactive iodine therapy, beta-blockers, thyroid surgery.", "diet": "Avoid iodine-rich foods (seaweed, iodized salt). Eat cruciferous vegetables, calcium-rich foods for bone health."},
    "Hypoglycemia": {"description": "Dangerously low blood sugar levels causing dizziness, confusion, sweating, and fainting.", "precautions": ["Never skip meals", "Always carry fast-acting glucose (glucose tablets, juice)", "Monitor blood sugar regularly", "Wear a medical alert bracelet"], "treatment": "IMMEDIATE: 15g fast-acting glucose (4oz juice, 3-4 glucose tablets). Follow with complex carbs + protein.", "diet": "Regular small meals every 3-4 hours. Always combine carbs with protein. Avoid alcohol on empty stomach."},
    "Osteoarthristis": {"description": "Degenerative joint disease where protective cartilage breaks down, causing pain and stiffness.", "precautions": ["Maintain healthy BMI to reduce joint load", "Low-impact exercise (swimming, cycling)", "Use joint protection techniques", "Apply heat/cold therapy for pain relief"], "treatment": "NSAIDs, physical therapy, corticosteroid injections, hyaluronic acid injections, joint replacement.", "diet": "Anti-inflammatory: omega-3 (fish, walnuts), turmeric, ginger, vitamin D and calcium for bone health."},
    "Arthritis": {"description": "Inflammation of one or more joints causing pain, swelling, stiffness, and reduced range of motion.", "precautions": ["Stay active with gentle exercise (yoga, walking)", "Protect joints during activities", "Maintain healthy weight", "Apply hot/cold packs for symptom relief"], "treatment": "NSAIDs, DMARDs for rheumatoid arthritis, corticosteroids, biologic agents, physical therapy.", "diet": "Anti-inflammatory diet: fatty fish, nuts, olive oil, colorful fruits and vegetables. Avoid processed foods."},
    "(vertigo) Paroymsal  Positional Vertigo": {"description": "Brief episodes of intense dizziness and spinning sensation triggered by specific head movements.", "precautions": ["Move head and body slowly and deliberately", "Install grab bars in bathroom", "Sleep with head slightly elevated", "Avoid sudden position changes (lying down to standing)"], "treatment": "Epley maneuver (canalith repositioning procedure), vestibular rehabilitation exercises.", "diet": "Low sodium diet reduces fluid retention in inner ear. Stay hydrated. Avoid caffeine and alcohol."},
    "Acne": {"description": "Common skin condition where hair follicles become clogged with oil and dead skin cells.", "precautions": ["Wash face gently twice daily with mild cleanser", "Never pop or squeeze pimples", "Use only non-comedogenic (oil-free) products", "Change pillowcases twice a week"], "treatment": "Topical retinoids, benzoyl peroxide, salicylic acid, antibiotics, isotretinoin for severe cystic acne.", "diet": "Low glycemic index diet, reduce dairy intake, eat zinc-rich foods (pumpkin seeds, lentils), omega-3."},
    "Urinary tract infection": {"description": "Bacterial infection in the urinary system (bladder, urethra, kidneys) causing burning and urgency.", "precautions": ["Drink 8-10 glasses of water daily", "Always wipe front to back", "Urinate shortly after intercourse", "Avoid holding urine for long periods"], "treatment": "Antibiotics (trimethoprim, nitrofurantoin, ciprofloxacin) for 3-7 days; increased fluid intake.", "diet": "Drink plenty of water and unsweetened cranberry juice. Avoid caffeine, alcohol, and spicy foods."},
    "Psoriasis": {"description": "Chronic autoimmune skin condition causing rapid skin cell buildup resulting in scales and red patches.", "precautions": ["Moisturize skin daily especially after bathing", "Identify and avoid personal triggers (stress, infections)", "Use gentle, fragrance-free skin products", "Protect skin from cuts and injuries"], "treatment": "Topical corticosteroids, vitamin D analogues, biologics (adalimumab), methotrexate, phototherapy.", "diet": "Anti-inflammatory diet: omega-3 rich fish, avoid alcohol (major trigger), reduce red meat."},
    "Impetigo": {"description": "Highly contagious superficial bacterial skin infection causing honey-colored crusting sores.", "precautions": ["Keep sores loosely covered with gauze", "Wash hands frequently with soap", "Don't share towels, clothing, or razors", "Keep nails short to avoid scratching and spreading"], "treatment": "Topical mupirocin antibiotic cream for mild cases; oral antibiotics (amoxicillin) for widespread infection.", "diet": "Vitamin C-rich foods to support skin healing. Adequate protein for immune and tissue repair."},
}

# Save it
with open("data/disease_info.json", "w") as f:
    json.dump(DISEASE_INFO, f, indent=2)

ok(f"data/disease_info.json  →  {WHITE}{len(DISEASE_INFO)}{RESET} diseases with full descriptions")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print(f"""
{GREEN}╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   🎉  Training Complete!                                         ║
║                                                                  ║
║   Best Model  :  {WHITE}{best_name:<47}{GREEN}║
║   Accuracy    :  {WHITE}{best_acc*100:.2f}%{GREEN}                                         ║
║   Diseases    :  {WHITE}{len(le.classes_)}{GREEN}  unique diseases                              ║
║   Symptoms    :  {WHITE}{len(ALL_SYMPTOMS)}{GREEN}  unique symptoms                             ║
║                                                                  ║
║   {WHITE}▶  Launch the app:  streamlit run app.py{GREEN}                    ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝{RESET}
""")