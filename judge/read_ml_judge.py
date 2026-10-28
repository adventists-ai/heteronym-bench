"""Reading judges for the EN/JA heteronym sets (data/readset_en_v2.json / data/readset_ja_v2.json). Given 16 kHz audio + one item, score every candidate reading
and return {label: score}; the argmax is the judged reading.
  whisper : Whisper-large-v3-turbo teacher-forced log-likelihood of the whole sentence with the target word rewritten per candidate
            (JA: target -> hiragana reading; EN: target -> respelling, e.g. read -> "reed"/"red").
  ctc     : (EN) wav2vec2-lv-60-espeak-cv-ft phoneme CTC log-likelihood of the sentence phone string (espeak per word) with the target's
            phones replaced by each candidate's IPA.
  ctccrop : (EN) context-free: forced-align the sentence, crop the waveform around the target word (+-60 ms) and score the candidate phones
            on the crop only (the phoneme model never hears the context, so it cannot "guess" from it).
  ctcfuse : (EN, default) ctc + ctccrop summed — best on phoneme-exact Kokoro audio and least biased toward the context-correct reading.
  ctcwin  : (EN) same CTC model, but the candidate phones are scored only inside a window around the target (context words +-2).
usage as a library:  J = Judge(methods=("ctc","whisper")); J.score(wav16k_np, item, words, lang, method)"""
import os, sys, re, json, numpy as np, torch, torch.nn.functional as F
WHISPER = os.environ.get("RM_WHISPER", "openai/whisper-large-v3-turbo"); W2V = os.environ.get("RM_W2V", "facebook/wav2vec2-lv-60-espeak-cv-ft")
def kata2hira(s): return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)

class Judge:
    def __init__(self, methods=("whisper", "ctc"), device="cuda"):
        self.dev = device; self.m = set(methods)
        if "whisper" in self.m:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            self.wp = WhisperProcessor.from_pretrained(WHISPER); self.wm = WhisperForConditionalGeneration.from_pretrained(WHISPER, torch_dtype=torch.float16).to(device).eval()
            tk = self.wp.tokenizer; self.sot = [tk.convert_tokens_to_ids(t) for t in ("<|startoftranscript|>",)]; self.eot = tk.convert_tokens_to_ids("<|endoftext|>")
            self.lang_id = {l: tk.convert_tokens_to_ids(f"<|{l}|>") for l in ("en", "ja")}; self.tr = tk.convert_tokens_to_ids("<|transcribe|>"); self.nots = tk.convert_tokens_to_ids("<|notimestamps|>")
        if self.m & {"ctc", "ctcwin", "ctccrop", "ctccrop1", "ctcfuse"}:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
            import espeakng_loader
            from phonemizer.backend.espeak.wrapper import EspeakWrapper
            EspeakWrapper.set_library(espeakng_loader.get_library_path()); os.environ["ESPEAK_DATA_PATH"] = espeakng_loader.get_data_path()
            from phonemizer.backend import EspeakBackend
            self.esp = EspeakBackend("en-us", with_stress=False, preserve_punctuation=False)
            self.fe = Wav2Vec2FeatureExtractor.from_pretrained(W2V); self.cm = Wav2Vec2ForCTC.from_pretrained(W2V).to(device).eval()
            self.vocab = json.load(open(W2V + "/vocab.json" if os.path.isdir(W2V) else __import__("huggingface_hub").hf_hub_download(W2V, "vocab.json"))); self.vk = sorted(self.vocab, key=len, reverse=True); self.ph_cache = {}

    # ---------------- whisper ----------------
    @torch.no_grad()
    def whisper_ll(self, wav, texts, lang):
        x = self.wp.feature_extractor(wav, sampling_rate=16000, return_tensors="pt").input_features.to(self.dev, torch.float16)
        enc = self.wm.model.encoder(x).last_hidden_state
        pre = self.sot + [self.lang_id[lang], self.tr, self.nots]
        seqs = [pre + self.wp.tokenizer((" " + t) if lang == "en" else t, add_special_tokens=False).input_ids + [self.eot] for t in texts]
        L = max(len(s) for s in seqs); ids = torch.full((len(seqs), L), self.eot, dtype=torch.long); msk = torch.zeros(len(seqs), L)
        for i, s in enumerate(seqs): ids[i, :len(s)] = torch.tensor(s); msk[i, len(pre):len(s)] = 1
        ids = ids.to(self.dev); out = self.wm.model.decoder(input_ids=ids[:, :-1], encoder_hidden_states=enc.expand(len(seqs), -1, -1)).last_hidden_state
        lp = self.wm.proj_out(out).float().log_softmax(-1).gather(-1, ids[:, 1:, None])[..., 0]
        return (lp * msk[:, 1:].to(self.dev)).sum(1).tolist()
    def cand_texts(self, it, words, lang):
        s, e = it["pos"]; t = it["text"]; rd = words[it["word"]]["readings"]; out = {}
        for lab in it["cands"]:
            if lang == "ja": out[lab] = [t[:s] + v + t[e:] for v in rd[lab]["kana"]]
            else: out[lab] = [t[:s] + rd[lab]["resp"] + t[e:]]
        return out

    # ---------------- phoneme CTC ----------------
    def ipa_tok(self, s):
        s = s.replace("ˈ", "").replace("ˌ", "").replace("ɡ", "ɡ").replace("g", "ɡ"); out = []; i = 0
        while i < len(s):
            for k in self.vk:
                if s.startswith(k, i) and k not in ("<s>", "</s>", "<pad>", "<unk>"): out.append(k); i += len(k); break
            else: i += 1   # unknown char (space, length mark leftovers ...)
        return out
    def word_ph(self, w):
        if w not in self.ph_cache: self.ph_cache[w] = self.ipa_tok(self.esp.phonemize([w], strip=True)[0])
        return self.ph_cache[w]
    def sent_phones(self, it, rd_ipa):
        """phone tokens of the sentence with the target word replaced by rd_ipa; also the (start,end) index of the target phones"""
        s, e = it["pos"]; t = it["text"]
        pre = re.findall(r"[A-Za-z']+", t[:s]); suf = re.findall(r"[A-Za-z']+", t[e:])
        P = [p for w in pre for p in self.word_ph(w)]; Q = [p for w in suf for p in self.word_ph(w)]; T = self.ipa_tok(rd_ipa)
        return P + T + Q, (len(P), len(P) + len(T)), pre, suf
    @torch.no_grad()
    def ctc_lp(self, wav):
        x = self.fe(wav, sampling_rate=16000, return_tensors="pt").input_values.to(self.dev)
        return self.cm(x).logits[0].float().log_softmax(-1)
    def ctc_ll(self, lp, phones):
        tgt = torch.tensor([self.vocab[p] for p in phones], device=lp.device)
        return -F.ctc_loss(lp[:, None], tgt[None], torch.tensor([lp.shape[0]]), torch.tensor([len(tgt)]), blank=self.vocab["<pad>"], reduction="sum", zero_infinity=True).item()

    # ---------------- public ----------------
    def score(self, wav, it, words, lang, method, lp=None):
        rd = words[it["word"]]["readings"]
        if method == "whisper":
            ct = self.cand_texts(it, words, lang); flat = [(lab, x) for lab, xs in ct.items() for x in xs]
            ll = self.whisper_ll(wav, [x for _, x in flat], lang); sc = {}
            for (lab, _), v in zip(flat, ll): sc[lab] = max(sc.get(lab, -1e9), v)
            return sc
        if method in ("ctc", "ctcwin"):
            assert lang == "en"
            lp = self.ctc_lp(wav) if lp is None else lp
            if method == "ctc": return {lab: self.ctc_ll(lp, self.sent_phones(it, rd[lab]["ipa"])[0]) for lab in it["cands"]}
            # window: +-2 context words around the target, scored with free (blank-or-anything) edges via a 'garbage' trick: we run CTC on the
            # sub-window of frames found by aligning the full default sentence, then score candidates on that segment only.
            ph0, (a, b), pre, suf = self.sent_phones(it, rd[it["cands"][0]]["ipa"])
            al = self.align(lp, ph0); n = len(ph0)
            ctxL = sum(len(self.word_ph(w)) for w in pre[-2:]); ctxR = sum(len(self.word_ph(w)) for w in suf[:2])
            i0, i1 = max(0, a - ctxL), min(n, b + ctxR); f0 = al[i0][0] if i0 < n else lp.shape[0] - 1; f1 = al[i1 - 1][1] if i1 > 0 else lp.shape[0]
            f0 = max(0, f0 - 3); f1 = min(lp.shape[0], f1 + 3); seg = lp[f0:f1]; sc = {}
            for lab in it["cands"]:
                ph, (a2, b2), _, _ = self.sent_phones(it, rd[lab]["ipa"]); sc[lab] = self.ctc_ll(seg, ph[i0: b2 + (i1 - b)])
            return sc
        if method == "ctcfuse":   # default EN judge: full-sentence CTC + context-free crop CTC (sum of log-likelihoods)
            lp = self.ctc_lp(wav) if lp is None else lp
            a, b = self.score(wav, it, words, lang, "ctc", lp=lp), self.score(wav, it, words, lang, "ctccrop", lp=lp)
            return {k: a[k] + b[k] for k in a}
        if method in ("ctccrop", "ctccrop1"):
            # context-free(ish): locate the target word by forced alignment of the whole sentence (union over candidates), then re-run the
            # phoneme model on the cropped waveform only, so its self-attention cannot use the sentence context to "guess" the reading.
            # ctccrop: target word only (+-RM_PAD s); ctccrop1: target plus one neighbouring word on each side (scored with their phones).
            assert lang == "en"
            nb = 1 if method == "ctccrop1" else 0
            lp = self.ctc_lp(wav) if lp is None else lp; nF = lp.shape[0]; hop = len(wav) / nF; f0, f1 = nF, 0
            for lab in it["cands"]:
                ph, (a, b), pre, suf = self.sent_phones(it, rd[lab]["ipa"]); al = self.align(lp, ph)
                pa = a - sum(len(self.word_ph(w)) for w in pre[len(pre) - nb:]) if nb else a
                pb = b + sum(len(self.word_ph(w)) for w in suf[:nb]) if nb else b
                f0 = min(f0, al[pa][0]); f1 = max(f1, al[pb - 1][1] + 1)
            pad = float(os.environ.get("RM_PAD", "0.06")); s0 = max(0, int(f0 * hop - pad * 16000)); s1 = min(len(wav), int(f1 * hop + pad * 16000))
            seg = wav[s0:s1]
            if len(seg) < 4000: seg = np.pad(seg, (0, 4000 - len(seg)))
            lq = self.ctc_lp(seg)
            L = [p for w in pre[len(pre) - nb:] for p in self.word_ph(w)] if nb else []; Rr = [p for w in suf[:nb] for p in self.word_ph(w)] if nb else []
            return {lab: self.ctc_ll(lq, L + self.ipa_tok(rd[lab]["ipa"]) + Rr) for lab in it["cands"]}
        raise ValueError(method)
    def align(self, lp, phones):
        """Viterbi CTC forced alignment -> list of (first_frame, last_frame) per phone"""
        import torchaudio.functional as TAF
        tgt = torch.tensor([[self.vocab[p] for p in phones]], device=lp.device)
        al, _ = TAF.forced_align(lp[None], tgt, blank=self.vocab["<pad>"]); al = al[0].tolist(); spans = []; j = -1; prev = self.vocab["<pad>"]
        for f, tkn in enumerate(al):
            if tkn != self.vocab["<pad>"] and (tkn != prev or prev == self.vocab["<pad>"]): j += 1; spans.append([f, f])
            elif tkn != self.vocab["<pad>"]: spans[j][1] = f
            prev = tkn
        while len(spans) < len(phones): spans.append([lp.shape[0] - 1, lp.shape[0] - 1])
        return spans
