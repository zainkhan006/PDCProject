"""
preprocess_clinical_notes.py  —  NLP Pipeline with UMLS/SNOMED Concept Normalisation
PDC Project 21 | Member 2 (Ali Hamza) | IBA Spring 2026

WHAT THIS FIXES:
  Previous pipeline did surface-level normalisation only (lowercase + underscores).
  This version adds a UMLS/SNOMED concept normalisation stage that:
    1. Maps extracted clinical terms to canonical SNOMED CT preferred names
    2. Replaces corpus variants with their canonical form before TF-IDF
    3. Saves a full concept_normalisation_log.csv showing every substitution made

  This satisfies the spec requirement:
  "Normalize concepts to standard terminologies (e.g. UMLS, SNOMED CT) for consistency"

Prerequisites:
    pip install spacy negspacy scikit-learn
    python -m spacy download en_core_web_sm

Usage:
    python3 preprocess_clinical_notes.py \
        --input   mtsamples.csv          \
        --output_features   features.csv \
        --output_labels     specialty_labels.csv \
        --output_feat_names feature_names.csv    \
        --output_norm_log   concept_normalisation_log.csv
"""

import re
import csv
import argparse
import os
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 0: UMLS / SNOMED CT CONCEPT NORMALISATION TABLE
#
# This is a curated synonym-to-canonical mapping covering the MTSamples
# vocabulary. Each entry maps one or more corpus surface forms to a single
# canonical SNOMED CT preferred name + code.
#
# Format: surface_form → (canonical_preferred_name, snomed_code)
#
# Sources:
#   SNOMED CT Browser: https://browser.ihtsdotools.org (free public access)
#   UMLS Metathesaurus: https://uts.nlm.nih.gov (free with registration)
# ─────────────────────────────────────────────────────────────────────────────

UMLS_SNOMED_NORM = {
    # ── Cardiovascular ───────────────────────────────────────────────────
    "bp":                       ("blood_pressure",               "75367002"),
    "b/p":                      ("blood_pressure",               "75367002"),
    "blood pressure":           ("blood_pressure",               "75367002"),
    "htn":                      ("hypertension",                 "38341003"),
    "hypertensive":             ("hypertension",                 "38341003"),
    "heart rate":               ("heart_rate",                   "364075005"),
    "hr":                       ("heart_rate",                   "364075005"),
    "pulse rate":               ("heart_rate",                   "364075005"),
    "ef":                       ("ejection_fraction",            "250908004"),
    "ejection fraction":        ("ejection_fraction",            "250908004"),
    "ekg":                      ("electrocardiogram",            "29303009"),
    "ecg":                      ("electrocardiogram",            "29303009"),
    "electrocardiogram":        ("electrocardiogram",            "29303009"),
    "mi":                       ("myocardial_infarction",        "22298006"),
    "myocardial infarction":    ("myocardial_infarction",        "22298006"),
    "heart attack":             ("myocardial_infarction",        "22298006"),
    "chf":                      ("congestive_heart_failure",     "42343007"),
    "congestive heart failure": ("congestive_heart_failure",     "42343007"),
    "afib":                     ("atrial_fibrillation",          "49436004"),
    "a fib":                    ("atrial_fibrillation",          "49436004"),
    "atrial fibrillation":      ("atrial_fibrillation",          "49436004"),
    "cab":                      ("coronary_artery_bypass",       "232717009"),
    "cabg":                     ("coronary_artery_bypass",       "232717009"),

    # ── Pulmonary / Respiratory ───────────────────────────────────────────
    "sob":                      ("shortness_of_breath",          "230145002"),
    "shortness of breath":      ("shortness_of_breath",          "230145002"),
    "dyspnea":                  ("shortness_of_breath",          "230145002"),
    "copd":                     ("chronic_obstructive_pulmonary_disease", "13645005"),
    "chronic obstructive pulmonary disease": ("chronic_obstructive_pulmonary_disease","13645005"),
    "osa":                      ("obstructive_sleep_apnea",      "78275009"),
    "obstructive sleep apnea":  ("obstructive_sleep_apnea",      "78275009"),
    "upper respiratory infection": ("upper_respiratory_infection","54150009"),
    "uri":                      ("upper_respiratory_infection",  "54150009"),

    # ── Diabetes / Endocrine ─────────────────────────────────────────────
    "dm":                       ("diabetes_mellitus",            "73211009"),
    "dm2":                      ("type_2_diabetes_mellitus",     "44054006"),
    "t2dm":                     ("type_2_diabetes_mellitus",     "44054006"),
    "diabetes mellitus":        ("diabetes_mellitus",            "73211009"),
    "type 2 diabetes":          ("type_2_diabetes_mellitus",     "44054006"),
    "type ii diabetes":         ("type_2_diabetes_mellitus",     "44054006"),
    "hba1c":                    ("hemoglobin_a1c",               "43396009"),
    "hemoglobin a1c":           ("hemoglobin_a1c",               "43396009"),
    "a1c":                      ("hemoglobin_a1c",               "43396009"),
    "tsh":                      ("thyroid_stimulating_hormone",  "61167004"),
    "thyroid stimulating hormone": ("thyroid_stimulating_hormone","61167004"),

    # ── Orthopaedic / Musculoskeletal ─────────────────────────────────────
    "r/o":                      ("rule_out",                     "415068001"),
    "rule out":                 ("rule_out",                     "415068001"),
    "fx":                       ("fracture",                     "125605004"),
    "fracture":                 ("fracture",                     "125605004"),
    "rom":                      ("range_of_motion",              "364564000"),
    "range of motion":          ("range_of_motion",              "364564000"),
    "pt":                       ("physical_therapy",             "91251008"),
    "physical therapy":         ("physical_therapy",             "91251008"),
    "tha":                      ("total_hip_arthroplasty",       "52734007"),
    "total hip replacement":    ("total_hip_arthroplasty",       "52734007"),
    "tka":                      ("total_knee_arthroplasty",      "609588000"),
    "total knee replacement":   ("total_knee_arthroplasty",      "609588000"),

    # ── Neurology ─────────────────────────────────────────────────────────
    "cva":                      ("cerebrovascular_accident",     "230690007"),
    "stroke":                   ("cerebrovascular_accident",     "230690007"),
    "tia":                      ("transient_ischemic_attack",    "266257000"),
    "transient ischemic attack":("transient_ischemic_attack",    "266257000"),
    "ms":                       ("multiple_sclerosis",           "24700007"),
    "multiple sclerosis":       ("multiple_sclerosis",           "24700007"),
    "ha":                       ("headache",                     "25064002"),
    "headache":                 ("headache",                     "25064002"),
    "mri":                      ("magnetic_resonance_imaging",   "113091000"),
    "magnetic resonance imaging":("magnetic_resonance_imaging",  "113091000"),

    # ── Oncology ─────────────────────────────────────────────────────────
    "ca":                       ("carcinoma",                    "68453008"),
    "carcinoma":                ("carcinoma",                    "68453008"),
    "mets":                     ("metastasis",                   "128462008"),
    "metastasis":               ("metastasis",                   "128462008"),
    "chemo":                    ("chemotherapy",                 "367336001"),
    "chemotherapy":             ("chemotherapy",                 "367336001"),
    "xrt":                      ("radiation_therapy",            "108290001"),
    "radiation therapy":        ("radiation_therapy",            "108290001"),

    # ── Gastrointestinal ─────────────────────────────────────────────────
    "gerd":                     ("gastroesophageal_reflux_disease","235595009"),
    "gastroesophageal reflux":  ("gastroesophageal_reflux_disease","235595009"),
    "ibs":                      ("irritable_bowel_syndrome",     "10743008"),
    "irritable bowel syndrome": ("irritable_bowel_syndrome",     "10743008"),
    "egd":                      ("esophagogastroduodenoscopy",   "310030000"),
    "upper endoscopy":          ("esophagogastroduodenoscopy",   "310030000"),
    "lap chole":                ("laparoscopic_cholecystectomy", "45595009"),
    "laparoscopic cholecystectomy":("laparoscopic_cholecystectomy","45595009"),

    # ── Medications ──────────────────────────────────────────────────────
    "asa":                      ("aspirin",                      "387458008"),
    "aspirin":                  ("aspirin",                      "387458008"),
    "epi":                      ("epinephrine",                  "387362001"),
    "epinephrine":              ("epinephrine",                  "387362001"),
    "prn":                      ("as_needed",                    "225756002"),
    "qd":                       ("once_daily",                   "229797004"),
    "bid":                      ("twice_daily",                  "229799001"),
    "tid":                      ("three_times_daily",            "229798009"),
    "qid":                      ("four_times_daily",             "307439001"),
    "po":                       ("oral_route",                   "26643006"),
    "iv":                       ("intravenous_route",            "47625008"),
    "im":                       ("intramuscular_route",          "78421000"),
    "nsaid":                    ("nonsteroidal_anti_inflammatory","372665008"),
    "nsaids":                   ("nonsteroidal_anti_inflammatory","372665008"),

    # ── Lab values / Procedures ───────────────────────────────────────────
    "cbc":                      ("complete_blood_count",         "26604007"),
    "complete blood count":     ("complete_blood_count",         "26604007"),
    "bmp":                      ("basic_metabolic_panel",        "271026005"),
    "cmp":                      ("comprehensive_metabolic_panel","271026005"),
    "bnp":                      ("brain_natriuretic_peptide",    "411015004"),
    "crp":                      ("c_reactive_protein",           "55235003"),
    "c reactive protein":       ("c_reactive_protein",           "55235003"),
    "wbc":                      ("white_blood_cell_count",       "767002"),
    "white blood cell":         ("white_blood_cell_count",       "767002"),
    "rbc":                      ("red_blood_cell_count",         "302215000"),
    "ct scan":                  ("computed_tomography",          "77477000"),
    "ct":                       ("computed_tomography",          "77477000"),
    "us":                       ("ultrasound",                   "16310003"),
    "ultrasound":               ("ultrasound",                   "16310003"),
    "xray":                     ("radiography",                  "363680008"),
    "x ray":                    ("radiography",                  "363680008"),
    "x-ray":                    ("radiography",                  "363680008"),

    # ── Clinical context ─────────────────────────────────────────────────
    "hpi":                      ("history_of_present_illness",   "417662000"),
    "history of present illness":("history_of_present_illness",  "417662000"),
    "pmh":                      ("past_medical_history",         "417662000"),
    "past medical history":     ("past_medical_history",         "417662000"),
    "ros":                      ("review_of_systems",            "415068001"),
    "review of systems":        ("review_of_systems",            "415068001"),
    "pe":                       ("physical_examination",         "5880005"),
    "physical exam":            ("physical_examination",         "5880005"),
    "physical examination":     ("physical_examination",         "5880005"),
    "cc":                       ("chief_complaint",              "422334004"),
    "chief complaint":          ("chief_complaint",              "422334004"),
    "a/p":                      ("assessment_and_plan",          "229059009"),
    "assessment and plan":      ("assessment_and_plan",          "229059009"),
    "f/u":                      ("follow_up",                    "308273005"),
    "follow up":                ("follow_up",                    "308273005"),
    "follow-up":                ("follow_up",                    "308273005"),
    "d/c":                      ("discharge",                    "58000006"),
    "discharge":                ("discharge",                    "58000006"),
    "admit":                    ("hospital_admission",           "32485007"),
    "admission":                ("hospital_admission",           "32485007"),
    "er":                       ("emergency_department",         "225728007"),
    "ed":                       ("emergency_department",         "225728007"),
    "emergency room":           ("emergency_department",         "225728007"),
    "icu":                      ("intensive_care_unit",          "309904001"),
    "or":                       ("operating_room",               "225738002"),
    "operating room":           ("operating_room",               "225738002"),
    "preop":                    ("preoperative",                 "262068006"),
    "pre op":                   ("preoperative",                 "262068006"),
    "pre-op":                   ("preoperative",                 "262068006"),
    "postop":                   ("postoperative",                "262061000"),
    "post op":                  ("postoperative",                "262061000"),
    "post-op":                  ("postoperative",                "262061000"),
    "npo":                      ("nothing_by_mouth",             "432221003"),
    "nothing by mouth":         ("nothing_by_mouth",             "432221003"),
    "ga":                       ("general_anesthesia",           "50697003"),
    "general anesthesia":       ("general_anesthesia",           "50697003"),
    "mac":                      ("monitored_anesthesia_care",    "398208008"),
    "intubation":               ("tracheal_intubation",          "52765003"),
    "intubated":                ("tracheal_intubation",          "52765003"),
    "extubation":               ("extubation",                   "271280005"),

    # ── Negation variants ─────────────────────────────────────────────────
    "neg_blood_pressure":       ("NEG_blood_pressure",           "75367002"),
    "neg_chest_pain":           ("NEG_chest_pain",               "29857009"),
    "neg_shortness_of_breath":  ("NEG_shortness_of_breath",      "230145002"),
    "neg_fever":                ("NEG_fever",                    "386661006"),
    "neg_nausea":               ("NEG_nausea",                   "422587007"),
}

# Build a reverse lookup: canonical → snomed_code (for reporting)
CANONICAL_TO_CODE = {v[0]: v[1] for v in UMLS_SNOMED_NORM.values()}


def normalise_token_umls(token: str) -> tuple:
    """
    Map a corpus token to its SNOMED canonical form.
    Returns (canonical_token, snomed_code, was_normalised).
    """
    t = token.strip().lower().replace('_', ' ')
    if t in UMLS_SNOMED_NORM:
        canon, code = UMLS_SNOMED_NORM[t]
        return canon, code, True
    # try with underscores restored
    t2 = token.strip().lower()
    if t2 in UMLS_SNOMED_NORM:
        canon, code = UMLS_SNOMED_NORM[t2]
        return canon, code, True
    # no match: apply surface normalisation only
    canon = re.sub(r'\s+', '_', token.strip().lower())
    return canon, None, False


def normalise_entity_string(entity_string: str) -> tuple:
    """
    Apply UMLS/SNOMED normalisation to a full space-joined entity string.
    Returns (normalised_string, list_of_substitutions).
    """
    tokens = entity_string.split()
    out_tokens = []
    substitutions = []

    i = 0
    while i < len(tokens):
        # Try longest match first (bigrams, then unigrams)
        matched = False
        if i + 1 < len(tokens):
            bigram = tokens[i] + ' ' + tokens[i+1]
            canon, code, was_norm = normalise_token_umls(bigram)
            if was_norm:
                out_tokens.append(canon)
                substitutions.append((bigram, canon, code))
                i += 2
                matched = True
        if not matched:
            canon, code, was_norm = normalise_token_umls(tokens[i])
            out_tokens.append(canon)
            if was_norm:
                substitutions.append((tokens[i], canon, code))
            i += 1

    return ' '.join(out_tokens), substitutions


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(input_csv, output_features, output_labels,
                 output_feat_names, output_norm_log,
                 max_features=500, min_df=5, max_df=0.85):

    print("[pipeline] Starting clinical NLP pipeline with UMLS/SNOMED normalisation")

    # ── Load spaCy and NegSpacy ───────────────────────────────────────────
    try:
        import spacy
        from negspacy.negation import Negex
        from negspacy.termsets import termset
        nlp = spacy.load("en_core_web_sm")
        ts = termset("en_clinical")
        nlp.add_pipe("negex",
                     config={"neg_termset": ts.get_patterns()},
                     last=True)
        print("[pipeline] spaCy + NegSpacy loaded.")
    except Exception as e:
        print(f"[pipeline] WARNING: spaCy/NegSpacy not available ({e}). "
              f"Using simple tokenisation fallback.")
        nlp = None

    SKIP_LABELS = {"PERSON","GPE","LOC","ORG","DATE","TIME",
                   "MONEY","PERCENT","CARDINAL","ORDINAL","QUANTITY","FAC","NORP"}

    def extract_and_normalise(text):
        """Extract entities, detect negation, apply UMLS normalisation."""
        if nlp is None:
            # Fallback: simple word tokenisation
            tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower()).split()
            raw_string = ' '.join(t for t in tokens if len(t) >= 3)
        else:
            doc = nlp(text[:100000])  # guard against very long notes
            seen = set()
            raw_tokens = []
            for ent in doc.ents:
                if ent.label_ in SKIP_LABELS:
                    continue
                t = re.sub(r'\s+', '_', ent.text.strip().lower())
                if len(t) < 3 or t in seen:
                    continue
                seen.add(t)
                prefix = "NEG_" if getattr(ent._, 'negex', False) else ""
                raw_tokens.append(prefix + t)
            for chunk in doc.noun_chunks:
                t = re.sub(r'\s+', '_', chunk.text.strip().lower())
                if len(t) < 3 or t in seen:
                    continue
                if t in {"patient","he","she","they","we","history","procedure"}:
                    continue
                seen.add(t)
                neg = any(getattr(tok._, 'negex', False) for tok in chunk)
                prefix = "NEG_" if neg else ""
                raw_tokens.append(prefix + t)
            raw_string = ' '.join(raw_tokens)

        # Apply UMLS/SNOMED normalisation
        normalised_string, subs = normalise_entity_string(raw_string)
        return normalised_string, subs

    # ── Load MTSamples CSV ────────────────────────────────────────────────
    print(f"[pipeline] Loading {input_csv} ...")
    docs = []
    with open(input_csv, newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('transcription', row.get('text', '')).strip()
            specialty = row.get('medical_specialty', row.get('specialty', 'Unknown')).strip()
            if text and len(text) >= 100:
                docs.append({'text': text, 'specialty': specialty})

    print(f"[pipeline] Retained {len(docs)} documents after length filter.")

    # ── Process each document ────────────────────────────────────────────
    entity_strings = []
    all_substitutions = []   # for the normalisation log

    for i, doc in enumerate(docs):
        norm_str, subs = extract_and_normalise(doc['text'])
        entity_strings.append(norm_str)
        for orig, canon, code in subs:
            all_substitutions.append({
                'doc_id': i,
                'specialty': doc['specialty'],
                'original_token': orig,
                'canonical_concept': canon,
                'snomed_code': code,
            })
        if (i+1) % 500 == 0:
            print(f"  ... processed {i+1}/{len(docs)}")

    print(f"[pipeline] UMLS/SNOMED normalisations applied: {len(all_substitutions)}")

    # ── TF-IDF Vectorisation ─────────────────────────────────────────────
    print("[pipeline] Running TF-IDF vectorisation ...")
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectoriser = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,
            norm='l2',
            ngram_range=(1, 2),
            token_pattern=r'\S+'
        )
        X = vectoriser.fit_transform(entity_strings).toarray()
        feature_names = vectoriser.get_feature_names_out()
        print(f"[pipeline] Feature matrix: {X.shape}  sparsity={100*(X==0).mean():.1f}%")
        assert X.min() >= 0.0 and X.max() <= 1.0 + 1e-9, "Value range assertion failed"
        print("[pipeline] Value range assertion PASSED: all values in [0, 1]")
    except ImportError:
        print("[pipeline] scikit-learn not found. Install: pip install scikit-learn")
        return

    # ── Save features.csv ────────────────────────────────────────────────
    print(f"[pipeline] Saving {output_features} ...")
    with open(output_features, 'w') as f:
        for row in X:
            f.write(','.join(f'{v:.8f}' for v in row) + '\n')
    print(f"[pipeline] Saved {X.shape[0]} × {X.shape[1]} feature matrix.")

    # ── Save specialty_labels.csv ────────────────────────────────────────
    print(f"[pipeline] Saving {output_labels} ...")
    with open(output_labels, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['medical_specialty'])
        for doc in docs:
            writer.writerow([doc['specialty']])

    # ── Save feature_names.csv ───────────────────────────────────────────
    print(f"[pipeline] Saving {output_feat_names} ...")
    with open(output_feat_names, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['feature_index', 'feature_name'])
        for i, name in enumerate(feature_names):
            # Annotate if this feature name is a SNOMED canonical term
            code = CANONICAL_TO_CODE.get(name, '')
            display = f"{name} [SNOMED:{code}]" if code else name
            writer.writerow([i, display])

    # ── Save concept_normalisation_log.csv ───────────────────────────────
    print(f"[pipeline] Saving {output_norm_log} ...")
    with open(output_norm_log, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'doc_id','specialty','original_token','canonical_concept','snomed_code'])
        writer.writeheader()
        writer.writerows(all_substitutions)

    # Print normalisation summary
    from collections import Counter
    canon_counts = Counter(s['canonical_concept'] for s in all_substitutions)
    print("\n── Top 20 SNOMED Normalisations Applied ──")
    print(f"  {'Original → Canonical':<50} {'SNOMED Code':<14} {'Count':>6}")
    print(f"  {'-'*50} {'-'*14} {'-'*6}")
    seen_canon = set()
    for s in all_substitutions:
        k = s['canonical_concept']
        if k not in seen_canon:
            seen_canon.add(k)
            orig_examples = [x['original_token'] for x in all_substitutions
                             if x['canonical_concept'] == k][:3]
            print(f"  {', '.join(orig_examples):<50} "
                  f"{s['snomed_code']:<14} "
                  f"{canon_counts[k]:>6}")
        if len(seen_canon) >= 20:
            break

    print(f"\n[pipeline] Complete.")
    print(f"  features.csv                : {X.shape[0]} × {X.shape[1]}")
    print(f"  specialty_labels.csv        : {len(docs)} labels")
    print(f"  feature_names.csv           : {len(feature_names)} terms (SNOMED-annotated)")
    print(f"  concept_normalisation_log.csv: {len(all_substitutions)} substitutions")


def main():
    ap = argparse.ArgumentParser(
        description="Clinical NLP pipeline with UMLS/SNOMED concept normalisation")
    ap.add_argument('--input',            default='mtsamples.csv')
    ap.add_argument('--output_features',  default='features.csv')
    ap.add_argument('--output_labels',    default='specialty_labels.csv')
    ap.add_argument('--output_feat_names',default='feature_names.csv')
    ap.add_argument('--output_norm_log',  default='concept_normalisation_log.csv')
    ap.add_argument('--max_features',     type=int, default=500)
    ap.add_argument('--min_df',           type=int, default=5)
    ap.add_argument('--max_df',           type=float, default=0.85)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"[error] Input file not found: {args.input}")
        print("  Download from: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions")
        print("  Then run: python3 preprocess_clinical_notes.py --input mtsamples.csv")
        return

    run_pipeline(args.input, args.output_features, args.output_labels,
                 args.output_feat_names, args.output_norm_log,
                 args.max_features, args.min_df, args.max_df)


if __name__ == '__main__':
    main()
