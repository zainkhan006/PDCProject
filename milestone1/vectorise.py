import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Configuration ──
INPUT_CSV    = "mtsamples.csv"
OUTPUT_CSV   = "features.csv"
MAX_FEATURES = 200     # number of TF-IDF features (columns)
MIN_DOC_LEN  = 100     # drop documents shorter than this (chars)

# ── Step 1: Load and clean ──
print("[vectorize] Loading", INPUT_CSV)
df = pd.read_csv(INPUT_CSV)

# Drop rows with no transcription
before = len(df)
df = df.dropna(subset=["transcription"])
print(f"[vectorize] Dropped {before - len(df)} rows with null transcription")

# Drop very short documents
before = len(df)
df = df[df["transcription"].str.len() >= MIN_DOC_LEN]
print(f"[vectorize] Dropped {before - len(df)} rows shorter than {MIN_DOC_LEN} chars")

print(f"[vectorize] Remaining documents: {len(df)}")

# ── Step 2: TF-IDF vectorization ──
print(f"[vectorize] Running TF-IDF with max_features={MAX_FEATURES}")

vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,   # keep top N terms by TF-IDF score
    stop_words="english",        # remove common words (the, is, and...)
    min_df=5,                    # term must appear in at least 5 docs
    max_df=0.95,                 # ignore terms in >95% of docs
    sublinear_tf=True            # apply log(1+tf) — standard for text
)

tfidf_matrix = vectorizer.fit_transform(df["transcription"])

# Convert sparse matrix to dense array
features = tfidf_matrix.toarray()

print(f"[vectorize] Feature matrix shape: {features.shape[0]} docs x {features.shape[1]} features")

# ── Step 3: Save as plain CSV (no header, no index) ──
pd.DataFrame(features).to_csv(OUTPUT_CSV, header=False, index=False)
print(f"[vectorize] Saved to {OUTPUT_CSV}")

# ── Print summary for report ──
N, F = features.shape
print(f"\n[vectorize] ── Summary ──")
print(f"  Documents (N) : {N}")
print(f"  Features  (F) : {F}")
print(f"  Top 10 terms  : {vectorizer.get_feature_names_out()[:10].tolist()}")
print(f"  Sparsity      : {(features == 0).sum() / features.size * 100:.1f}%")
print(f"\n[vectorize] To run FCM on this data:")
print(f"  ./fcm features.csv {N} {F} 4")