"""Heteronym reading accuracy for English / Japanese TTS audio.

usage:
  python eval_enja.py --lang ja --wav_dir out/ja [--out result.json] [--method whisper]
  python eval_enja.py --lang en --wav_dir out/en [--method ctcfuse]
Audio files must be named <item_id>.wav (item ids are the keys of data/readset_<lang>_v2.json["items"]).
Any sample rate / channel count is accepted (resampled to 16 kHz mono).
Judges: JA = Whisper-large-v3-turbo teacher-forced likelihood with the target word rewritten in each candidate kana;
        EN = wav2vec2-lv-60-espeak-cv-ft phoneme CTC, full sentence + context-free crop of the target word ('ctcfuse').
Model paths: env RM_WHISPER (default openai/whisper-large-v3-turbo), RM_W2V (default facebook/wav2vec2-lv-60-espeak-cv-ft)."""
import os, sys, json, argparse, collections, numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "judge"))
sys.path += [p for p in os.environ.get("RM_EXTRA_PATH", "").split(":") if p]   # optional: extra site dirs appended last
ap = argparse.ArgumentParser(); ap.add_argument("--lang", required=True, choices=["en", "ja"]); ap.add_argument("--wav_dir", required=True)
ap.add_argument("--method", default=None); ap.add_argument("--out", default=None); A = ap.parse_args()
RS = json.load(open(os.path.join(HERE, "data", f"readset_{A.lang}_v2.json"))); meth = A.method or ("whisper" if A.lang == "ja" else "ctcfuse")
REL = set(json.load(open(os.path.join(HERE, "validation", "judge_reliability.json"))).get(A.lang, {}).get("reliable", []))
from read_ml_judge import Judge
J = Judge(methods={meth}); P = {}; miss = 0
for k, it in sorted(RS["items"].items()):
    f = os.path.join(A.wav_dir, k + ".wav")
    if not os.path.exists(f): miss += 1; continue
    w, sr = sf.read(f, dtype="float32", always_2d=True); w = w.mean(1)
    if sr != 16000:
        import librosa; w = librosa.resample(w, orig_sr=sr, target_sr=16000)
    sc = J.score(w, it, RS["words"], A.lang, meth); pr = max(sc, key=sc.get)
    P[k] = dict(word=it["word"], answer=it["answer"], pred=pr, ok=pr == it["answer"], minority=it["minority"], heldout=it.get("hold", False), scores={a: round(b, 3) for a, b in sc.items()})
pct = lambda x: round(100 * float(np.mean(x)), 1) if x else None
v = list(P.values())
S = dict(lang=A.lang, judge=meth, n=len(v), missing=miss, acc=pct([p["ok"] for p in v]), acc_minority=pct([p["ok"] for p in v if p["minority"]]),
         acc_majority=pct([p["ok"] for p in v if not p["minority"]]), acc_reliable_words=pct([p["ok"] for p in v if p["word"] in REL]),
         pred_default_rate=pct([p["pred"] == RS["words"][p["word"]]["major"] for p in v]))
out = A.out or os.path.join(A.wav_dir.rstrip("/") + f"_readscore_{A.lang}.json")
json.dump(dict(summary=S, items=P), open(out, "w"), ensure_ascii=False, indent=1); print(json.dumps(S, ensure_ascii=False))
