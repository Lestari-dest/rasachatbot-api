# rasa_chatbot.py
import os, json, re, random, torch
import torch.nn.functional as F
from typing import Dict, Optional, List

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import google.generativeai as genai


class RasaChatbot:
    def __init__(self, model_name: str, gemini_api_key: str):
        print("v5 Minimal-postproc: keep original vibe, add crisis+pointer reco")
        self.model_name = model_name
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN", None)

        # === Load IndoBERT ===
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=hf_token)
        self.model.eval()

        # === Calibration (optional) ===
        calib_path = os.getenv("CALIB_JSON")
        self.calib = None
        if calib_path and os.path.exists(calib_path):
            with open(calib_path, "r") as f:
                self.calib = json.load(f)
        else:
            try:
                downloaded = hf_hub_download(
                    repo_id=model_name, repo_type="model",
                    filename=os.getenv("CALIB_JSON_HF", "calibration.json"),
                    use_auth_token=hf_token
                )
                with open(downloaded, "r") as f:
                    self.calib = json.load(f)
            except Exception:
                self.calib = {"labels":["anger","happy","sadness","love","fear"],"max_length":192,
                              "sadness_bias":0.0,"delta_ang":0.0,"delta_hap":0.0,"delta_fear":0.0,"pos_sad":0.0}

        self.labels = self.calib.get("labels", ["anger","happy","sadness","love","fear"])
        self.confidence_threshold = 0.65

        # === LLM ===
        genai.configure(api_key=gemini_api_key)
        self.chat_model = genai.GenerativeModel("gemini-2.0-flash")
        self.chat_session = self.chat_model.start_chat(history=[])

        # === Lightweight cues (tidak agresif) ===
        self.ANGER = [r'\bmarah\b', r'\bkesal\b', r'\bkesel\b', r'\bjengkel\b', r'\bbenci\b', r'\bemosi\b']
        self.SAD   = [r'\bsedih\b', r'\bkecewa\b', r'\bgalau\b', r'\bmurung\b', r'\bdown\b', r'\bputus asa\b', r'\bhampa\b']
        self.HAPPY = [r'\bbahagia\b', r'\bsenang\b', r'\bgembira\b', r'\blega\b', r'\bsemangat\b']
        self.FEAR  = [r'\btakut\b', r'\bcemas\b', r'\bkhawatir\b', r'\bdeg(-| )?degan\b', r'\bgentar\b']

        # === Memory & profile ===
        self.previous_sentiment = None
        self.turn_count = 0
        self.short_term_memory: List[Dict] = []
        self.user_personality_profile = {
            'preferred_pronouns': 'aku',   # 'aku'/'gue'/'saya'
            'emoji_usage': False,
            'exclamation_tendency': 'low',
            'openness_level': 1
        }

        # === RESPONSE STYLE (ringan) ===
        self.response_styles = {
            'sadness': {1:["Hmm, kayaknya lagi nggak enak."], 2:["Pasti berat ya rasanya."], 3:["Aku kebayang ini berat."]},
            'anger':   {1:["Pantes sih kalau kesel."],         2:["Wajar kok kalau kamu marah."], 3:["Aku ngerti kenapa kamu marah."]},
            'fear':    {1:["Deg-degan gitu ya."],              2:["Pasti bikin nggak tenang."],  3:["Aku paham kekhawatiran kamu."]},
            'happy':   {1:["Asik banget!"],                    2:["Seneng denger kabar baikmu."],3:["Ikut seneng dengernya."]},
            'love':    {1:["Kedengeran sayang banget."],       2:["So sweet ya rasanya."],       3:["Perasaanmu kerasa hangat."]},
            'neutral': {1:["I see."],                          2:["Menarik juga."],              3:["Oke, aku nangkep maksudmu."]}
        }

        # === Add-ons (RECO + CRISIS) — hanya post-proc, supaya vibe asli tetap ===
        self.last_reco_turn = -10
        self.reco_cooldown = 3                        # jeda minimal sebelum muncul lagi
        self.EXPLICIT_RECO_WORDS = [                  # user jelas minta
            r'\brekomendasi\b', r'\bsaran\b', r'\btips\b', r'\bartikel\b', r'\bkonten\b', r'\bvideo\b',
            r'\bbutuh bantuan\b', r'\bhotline\b', r'\bpsikolog\b',
            r'\bcari(kan)?\b.*\b(artikel|video|konten)\b', r'\b(artikel|video|konten)\b.*\bapa\b'
        ]
        # Krisis
        self.SELF_HARM = [
            r'\bbunuh diri\b', r'\bakhiri hidup\b', r'\bgak mau hidup\b', r'\bcapek hidup\b',
            r'\bmenyakiti diri\b', r'\bmelukai diri\b', r'\bnyakitin diri\b',
            r'\b(pengen|pingin|ingin|mau|pgn)\s*mati\b', r'\bmending\s*mati\b', r'\blebih baik\s*mati\b',
            r'\bmau\s*mati\s*aja\b', r'\baku\s*(pengen|mau)\s*mati\b'
        ]
        self.HARM_OTHERS = [r'\bmelukai orang\b', r'\bmenyakiti orang\b', r'\bunuh .*(dia|mereka)\b']
        self.ABUSE = [r'\bdipukul\b', r'\bkekerasan\b', r'\bKDRT\b', r'\bdilecehkan\b', r'\bpelecehan\b']

        self.HOTLINES_ID = (
            "Keselamatan dulu: jika kamu dalam bahaya/krisis hubungi 112 (darurat umum), 110 (Polisi), "
            "119 (PSC/ambulans), SEJIWA 119 ext. 8 (dukungan psikologis), atau SAPA 129 / WA 08111-129-129 "
            "(perlindungan perempuan & anak). Jika darurat sekarang juga, minta bantuan orang terdekat."
        )

    # ---------------- Utils ringkas ----------------
    def _flatten(self, s: str) -> str:
        if not isinstance(s, str): s = str(s)
        s = s.replace("\r"," ").replace("\n"," ")
        s = s.replace("**","").replace("*","").replace("•","").replace("—","-").replace('"','”')
        s = re.sub(r'\s+',' ', s)
        s = re.sub(r'\s+([,.;:!?])', r'\1', s)
        return s.strip()

    def _split_sentences(self, s: str) -> List[str]:
        parts = re.split(r'(?<=[.!?])\s+', s.strip())
        return [p for p in parts if p]

    def _similar(self, a: str, b: str) -> bool:
        if a == b: return True
        aw = " ".join(a.split()[:6]); bw = " ".join(b.split()[:6])
        if aw and aw == bw: return True
        wa, wb = set(a.split()), set(b.split())
        if len(wa) >= 3 and len(wb) >= 3:
            j = len(wa & wb) / max(1, len(wa | wb))
            if j > 0.75: return True
        return False

    def _dedupe(self, s: str) -> str:
        parts = self._split_sentences(s)
        out, seen = [], []
        for p in parts:
            norm = re.sub(r'[“”"]','', p.lower())
            norm = re.sub(r'\s+',' ', norm)
            if not any(self._similar(norm, re.sub(r'[“”"]','', q.lower())) for q in seen):
                out.append(p); seen.append(p)
        return " ".join(out)

    # ---------- Style ----------
    def analyze_user_style(self, user_input: str) -> Dict:
        st = {}
        if re.search(r'\b(gue|gw)\b', user_input.lower()): st['pronouns']='gue'
        elif re.search(r'\bsaya\b', user_input.lower()):    st['pronouns']='saya'
        elif re.search(r'\b(aku|ak)\b', user_input.lower()):st['pronouns']='aku'
        else:                                               st['pronouns']=self.user_personality_profile['preferred_pronouns']
        st['exclamation_level'] = 'high' if user_input.count('!')>=3 else ('medium' if user_input.count('!')>=1 else 'low')
        st['uses_repetition'] = bool(re.findall(r'(\w)\1{2,}', user_input.lower()))
        st['uses_emoji'] = bool(re.findall(r'[😀-🙏]', user_input))
        st['message_length'] = 'short' if len(user_input.split())<=3 else ('medium' if len(user_input.split())<=10 else 'long')
        return st

    def mirror_user_style(self, text: str, style: Dict) -> str:
        t = text
        pron = style.get('pronouns', self.user_personality_profile['preferred_pronouns'])
        if pron == 'gue':
            t = re.sub(r'\baku\b','gue', t)
            t = re.sub(r'\bkamu\b','lu', t); t = re.sub(r'\bKamu\b','Lu', t)
        elif pron == 'saya':
            t = re.sub(r'\baku\b','saya', t)
        ex = style.get('exclamation_level','low')
        if ex == 'high' and '.' in t: t = t.replace('.', '!')
        elif ex == 'medium' and t.count('!')==0 and t.endswith('.'): t = t[:-1]+'!'
        if style.get('uses_repetition'):
            t = re.sub(r'\bbanget\b','bangettt', t)
        return t

    def update_personality_profile(self, style: Dict):
        if style.get('pronouns'): self.user_personality_profile['preferred_pronouns']=style['pronouns']
        if style.get('uses_emoji'): self.user_personality_profile['emoji_usage']=True
        if self.turn_count>3 and style.get('message_length')=='long':
            self.user_personality_profile['openness_level']=min(5,self.user_personality_profile['openness_level']+1)

    # ---------- Sentiment ----------
    def _clean(self, text: str):
        t = text.lower()
        t = re.sub(r'http[s]?://\S+',' [URL] ', t)
        t = re.sub(r'[^\w\s!?.,:;()]+',' ', t)
        t = re.sub(r'\s+',' ', t).strip()
        return t

    def analyze_sentiment(self, text: str) -> Dict:
        try:
            t = self._clean(text)
            enc = self.tokenizer(
                t, return_tensors="pt", truncation=True, padding=True,
                max_length=int(self.calib.get("max_length",192))
            )
            with torch.no_grad():
                logits = self.model(**enc).logits.squeeze(0)
            # adjust sadness if ada
            try: sad_idx = self.labels.index("sadness")
            except ValueError: sad_idx = 2
            probs = F.softmax(logits, dim=-1)
            conf, idx = torch.max(probs, dim=0)
            if conf.item() < self.confidence_threshold:
                return {'dominant_sentiment':'neutral','confidence':conf.item(),'uncertain':True}
            return {'dominant_sentiment': self.labels[int(idx)], 'confidence': conf.item(), 'uncertain': False}
        except Exception as e:
            print("Sentiment error:", e)
            return {'dominant_sentiment':'neutral','confidence':0.5,'uncertain':True}

    # ---------- LLM prompt (ringan, supaya feel model pertama tetap) ----------
    def _is_question_or_task(self, text: str) -> bool:
        # jaga agar bisa Q&A, bukan cuma curhat
        t = text.lower()
        return ('?' in t) or any(w in t for w in ['apa','bagaimana','gimana','mengapa','kenapa','kapan','dimana','di mana','berapa','cara','tutorial'])

    def _prompt_support(self, user_input: str, sentiment: str) -> str:
        # super simple: biar LLM yang natural, tanpa aturan kaku
        return (
            f"Tulis balasan empatik dan natural dalam Bahasa Indonesia, 1 paragraf, tanpa bullet dan tanpa baris baru. "
            f"Jangan mengutip teks pengguna secara langsung. Pengguna berkata: {user_input!r}. "
            f"Perasaan pengguna terdeteksi: {sentiment}. Validasi perasaan, ajak bercerita seperlunya, jangan menggurui."
        )

    def _prompt_qa(self, user_input: str) -> str:
        return (
            f"Jawab pertanyaan ini secara jelas, ringkas, dan akurat dalam Bahasa Indonesia, 1 paragraf, tanpa bullet dan tanpa baris baru. "
            f"Hindari mengulang pertanyaan. Pertanyaan: {user_input!r}"
        )

    # ---------- Add-ons: crisis + pointer reco (post-proc) ----------
    def _explicit_reco_requested(self, text: str) -> bool:
        t = text.lower()
        return any(re.search(p, t) for p in self.EXPLICIT_RECO_WORDS)

    def _detect_crisis(self, text: str) -> Dict:
        t = text.lower()
        is_self_harm  = any(re.search(p, t) for p in self.SELF_HARM)
        is_harm_other = any(re.search(p, t) for p in self.HARM_OTHERS)
        is_abuse      = any(re.search(p, t) for p in self.ABUSE)
        is_crisis = is_self_harm or is_harm_other or is_abuse
        ctype = 'self_harm' if is_self_harm else ('harm_others' if is_harm_other else ('abuse' if is_abuse else None))
        return {'is_crisis': is_crisis, 'type': ctype}

    def _negative_streak(self, k: int = 2) -> bool:
        if len(self.short_term_memory) < k: return False
        neg = {'sadness','fear','anger'}
        return all(m['sentiment'] in neg for m in self.short_term_memory[-k:])

    def _should_offer_reco(self, user_input: str, sentiment: str) -> bool:
        # eksplisit → boleh kapan saja
        if self._explicit_reco_requested(user_input): return True
        # selain itu minimal turn ke-3
        if self.turn_count < 3: return False
        # anti spam
        if self.turn_count - self.last_reco_turn < self.reco_cooldown: return False
        # emosi negatif atau streak
        if sentiment in {'sadness','fear','anger'} or self._negative_streak(2):
            return True
        return False

    def _preface_reco(self, sentiment: str, asked: bool) -> str:
        soft = {'sadness':"Aku kebayang ini berat.", 'fear':"Deg-degan itu nggak enak.", 'anger':"Pantes sih kalau kesel.",
                'happy':"Seneng denger kabar baikmu.", 'love':"Kedengeran sayang banget.", 'neutral':"Biar bantuannya makin pas."}
        opener = soft.get(sentiment, soft['neutral'])
        ask = "Kalau kamu mau, aku punya saran kecil" if not asked else "Oke, aku share saran kecil"
        return f"{opener} {ask}: "

    def _build_reco_pointer(self, sentiment: str, asked: bool) -> str:
        bank = {
            'fear': ["teknik napas 4-4-4", "grounding 5-4-3-2-1", "langkah kecil atasi cemas"],
            'sadness': ["aktivitas mini 5 menit", "self-compassion langkah awal", "gratitude 3 hal"],
            'anger': ["time-out 90 detik", "I-message untuk ungkap emosi", "hal yang bisa kukontrol vs tidak"],
            'love': ["komunikasi asertif 2 kalimat", "boundaries sehat dalam relasi"],
            'happy': ["jurnal syukur 3 hal", "merayakan kemenangan kecil"],
            'neutral': ["wheel of emotion", "identifikasi kebutuhan diri sederhana"]
        }
        items = bank.get(sentiment, bank['neutral'])[:3]
        return self._preface_reco(sentiment, asked) + "coba cari tentang '" + "', '".join(items[:-1]) + "', atau '" + items[-1] + "'."

    def _build_hotline_inline(self) -> str:
        return (" Aku khawatir sama keselamatan kamu; kalau kamu merasa tidak aman sekarang, prioritasnya keselamatan ya. "
                + self.HOTLINES_ID + " Kamu aman sekarang?")

    def _apply_postproc(self, user_input: str, base_resp: str, sentiment: str, style: Dict) -> str:
        # 1) crisis always wins
        crisis = self._detect_crisis(user_input)
        if crisis['is_crisis']:
            # jangan ubah kalimat awal; tambahkan hotline di ekor
            out = base_resp
            out = out.split(".")[0] + "." + self._build_hotline_inline() if "." in out else out + "." + self._build_hotline_inline()
            out = self.mirror_user_style(out, style)
            out = self._flatten(self._dedupe(out))
            return out

        # 2) pointer reco (hanya jika perlu/minta, tidak di awal)
        asked = self._explicit_reco_requested(user_input)
        if self._should_offer_reco(user_input, sentiment):
            reco = self._build_reco_pointer(sentiment, asked)
            out = self.mirror_user_style(base_resp + " " + reco, style)
        else:
            out = self.mirror_user_style(base_resp, style)

        # 3) rapikan: 1 paragraf + dedupe
        out = self._flatten(self._dedupe(out))
        return out

    # ---------------- Main chat ----------------
    def chat(self, user_input: str) -> Dict:
        try:
            style = self.analyze_user_style(user_input)
            self.update_personality_profile(style)

            # Sentiment
            senti = self.analyze_sentiment(user_input)
            current_sentiment = senti['dominant_sentiment']

            # Pilih mode: support vs Q&A
            if self._is_question_or_task(user_input):
                prompt = self._prompt_qa(user_input)
            else:
                prompt = self._prompt_support(user_input, current_sentiment)

            # LLM
            resp = self.chat_session.send_message(prompt)
            base_text = resp.text.strip()

            # Post-proc (KRISIS/RECO) TANPA mengubah feel dasar
            final_text = self._apply_postproc(user_input, base_text, current_sentiment, style)

            # Memory & state
            self.short_term_memory.append({'user':user_input,'bot':final_text,'sentiment':current_sentiment,'turn':self.turn_count})
            if len(self.short_term_memory)>3: self.short_term_memory.pop(0)
            self.previous_sentiment = current_sentiment
            self.turn_count += 1

            return {
                'response': final_text,
                'sentiment': current_sentiment,
                'confidence': senti.get('confidence', 1.0),
                'transition': None,
                'empathy_level': 1,
                'style_analysis': style,
                'special_case': False
            }
        except Exception as e:
            print("Chat error:", e)
            return {'response': "Maaf, lagi ada gangguan. Coba lagi ya.", 'sentiment':'neutral',
                    'confidence':0.5,'transition':None,'empathy_level':1,'style_analysis':{},'special_case':False}

    def reset_chat_session(self):
        try:
            self.chat_session = self.chat_model.start_chat(history=[])
            self.turn_count = 0
            self.short_term_memory = []
            self.previous_sentiment = None
            self.user_personality_profile = {
                'preferred_pronouns': 'aku',
                'emoji_usage': False,
                'exclamation_tendency': 'low',
                'openness_level': 1
            }
            self.last_reco_turn = -10
            print("🔄 Session reset")
        except Exception as e:
            print("Reset error:", e)
