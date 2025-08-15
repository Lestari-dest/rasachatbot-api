# rasa_chatbot.py
import os, json, re, random, torch
import torch.nn.functional as F
from typing import Dict, Optional, List

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import google.generativeai as genai


class RasaChatbot:
    def __init__(self, model_name: str, gemini_api_key: str):
        print("v3-inline Initializing RasaChatbot (calibrated sentiment, crisis & inline reco, style-aware)…")
        self.model_name = model_name
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN", None)

        # 1) Load IndoBERT
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=hf_token)
        self.model.eval()

        # 2) Load calibration (local > HF > defaults)
        calib_path = os.getenv("CALIB_JSON")
        self.calib = None
        if calib_path and os.path.exists(calib_path):
            with open(calib_path, "r") as f:
                self.calib = json.load(f)
        else:
            fname = os.getenv("CALIB_JSON_HF", "calibration.json")
            try:
                downloaded = hf_hub_download(
                    repo_id=model_name, repo_type="model",
                    filename=fname, use_auth_token=hf_token
                )
                with open(downloaded, "r") as f:
                    self.calib = json.load(f)
            except Exception as e:
                print(f"⚠️ calibration.json not found on Hub ({fname}). Using defaults. Error: {e}")
                self.calib = {
                    "labels": ["anger","happy","sadness","love","fear"],
                    "sadness_bias": 0.0, "delta_ang": 0.0, "delta_hap": 0.0, "delta_fear": 0.0, "pos_sad": 0.0,
                    "max_length": 192
                }

        # 3) Labels
        self.labels = self.calib.get("labels", ["anger","happy","sadness","love","fear"])

        # 4) Gemini
        genai.configure(api_key=gemini_api_key)
        self.chat_model = genai.GenerativeModel("gemini-2.0-flash")
        self.chat_session = self.chat_model.start_chat(history=[])

        # 5) Cues & emoji
        self.ANGER = [r'\bmarah\b', r'\bkesal\b', r'\bkesel\b', r'\bjengkel\b', r'\bdongkol\b',
                      r'\bsebal\b', r'\bsebel\b', r'\bgeram\b', r'\bbenci\b', r'\bmurka\b', r'\bemosi\b', r'\bngamuk\b']
        self.SAD   = [r'\bsedih\b', r'\bkecewa\b', r'\bterpuruk\b', r'\bgalau\b', r'\bmurung\b', r'\bhancur\b',
                      r'\bpatah hati\b', r'\bdown\b', r'\bmenangis\b', r'\bnangis\b', r'\bduka(cita)?\b',
                      r'\bterluka\b', r'\bsunyi\b', r'\bkesepian\b', r'\bsepi\b', r'\bputus asa\b', r'\bhampa\b']
        self.HAPPY = [r'\bbahagia\b', r'\bsenang\b', r'\bgembira\b', r'\bsuka\b', r'\bceria\b',
                      r'\blega\b', r'\bpuas\b', r'\bsemangat\b', r'\bbersyukur\b']
        self.FEAR  = [r'\btakut\b', r'\bcemas\b', r'\bkhawatir\b', r'\bwas[- ]?was\b',
                      r'\bdeg(-| )?degan\b', r'\bparno\b', r'\bparanoid\b', r'\bgentar\b', r'\bketakutan\b', r'\bngeri\b']

        self.EMO_ANGER = set("😠😡🤬👿")
        self.EMO_SAD   = set("😢😭☹🙁😞😔")
        self.EMO_HAPPY = set("😀😃😄😁🙂😊🥰😍😺")
        self.EMO_FEAR  = set("😱😨😰")

        # 6) Memory & config
        self.previous_sentiment = None
        self.turn_count = 0
        self.short_term_memory: List[Dict] = []
        self.user_personality_profile = {
            'preferred_pronouns': 'aku',
            'formality_level': 'casual',
            'emoji_usage': False,
            'exclamation_tendency': 'low',
            'openness_level': 1
        }
        self.confidence_threshold = 0.65

        # 7) Response styles
        self.response_styles = {
            'sadness': {
                'level_1': ["Kedengeran berat ya.", "Hmm, kayaknya lagi nggak enak.", "Pasti susah ya."],
                'level_2': ["Kayaknya lagi susah banget ya. Mau cerita lebih lanjut?", "Pasti berat banget rasanya sekarang."],
                'level_3': ["Aku bisa ngebayangin betapa beratnya perasaan kamu. Kadang emang ada masa-masa kayak gini ya."]
            },
            'anger': {
                'level_1': ["Sampe kesel gitu ya!", "Hmm, bikin emosi banget.", "Pantes sih kalau kesel."],
                'level_2': ["Sampe segitunya ya keselnya. Ada apa sih tadi?", "Wajar banget kalau sampai marah gitu."],
                'level_3': ["Aku ngerti banget kenapa kamu marah. Kadang emang ada hal yang bikin kita nggak bisa sabar."]
            },
            'fear': {
                'level_1': ["Hmm, wajar sih kalau deg-degan.", "Bikin khawatir ya.", "Deg-degan gitu ya."],
                'level_2': ["Pasti bikin nggak tenang banget ya. Emang ada apa?", "Deg-degan kayak gini emang nggak enak."],
                'level_3': ["Kekhawatiran kayak gitu emang bikin susah ya. Aku paham banget perasaan kamu."]
            },
            'happy': {
                'level_1': ["Asik banget!", "Seneng deh denger kamu happy.", "Kedengeran seru ya."],
                'level_2': ["Seneng banget deh denger kamu udah lebih baik! Ada apa nih?", "Asik banget sih, pasti lega ya."],
                'level_3': ["Aku seneng banget liat kamu bahagia kayak gini. Cerita dong apa yang bikin happy."]
            },
            'love': {
                'level_1': ["Kedengeran sayang banget ya.", "Aww, sweet banget.", "Gemes deh."],
                'level_2': ["Kedengeran sayang banget sama dia. Gimana ceritanya?", "Aww, pasti bikin hati anget ya."],
                'level_3': ["Perasaan sayang kayak gini emang indah ya. Mau cerita lebih tentang dia?"]
            },
            'neutral': {
                'level_1': ["Hmm, gitu ya.", "Oh begitu.", "I see."],
                'level_2': ["Hmm, ada yang lagi kepikiran ya?", "Gimana perasaan kamu tentang itu?"],
                'level_3': ["Menarik juga. Aku penasaran sama pemikiran kamu tentang hal ini."]
            }
        }

        # 8) Rekomendasi & Krisis config
        self.last_reco_turn = -10
        self.reco_cooldown = 3
        self.EXPLICIT_RECO_WORDS = [
            r'\brekomendasi\b', r'\bsaran\b', r'\btips\b', r'\bartikel\b',
            r'\bbutuh bantuan\b', r'\bnomor darurat\b', r'\bhotline\b', r'\bpsikolog\b'
        ]
        self.SELF_HARM = [
            r'\bbunuh diri\b', r'\bakhiri hidup\b', r'\bgak mau hidup\b', r'\bcapek hidup\b',
            r'\bmenyakiti diri\b', r'\bmelukai diri\b', r'\bnyakitin diri\b'
        ]
        self.HARM_OTHERS = [r'\bmelukai orang\b', r'\bmenyakiti orang\b', r'\bunuh .*(dia|mereka)\b']
        self.ABUSE = [r'\bdipukul\b', r'\bkekerasan\b', r'\bKDRT\b', r'\bdilecehkan\b', r'\bpelecehan\b']

        self.HOTLINES_ID = (
            "Keselamatan dulu: jika kamu dalam bahaya/krisis hubungi 112 (darurat umum), 110 (Polisi), "
            "119 (PSC/ambulans), SEJIWA 119 ext. 8 (dukungan psikologis), atau SAPA 129 / WA 08111-129-129 "
            "(perlindungan perempuan & anak). Jika darurat sekarang juga, minta bantuan orang terdekat."
        )

        print("✅ RasaChatbot inline style-aware ready.")

    # ============ Helpers (cues & cleaning) ============
    def _has_emoji(self, s, charset): return any(ch in charset for ch in s)

    def _clean_and_cue(self, text: str):
        t = str(text).lower()
        emojis = ''.join([ch for ch in t if ch in (self.EMO_ANGER|self.EMO_SAD|self.EMO_HAPPY|self.EMO_FEAR)])
        t = re.sub(r'http[s]?://\S+', ' [URL] ', t)
        t = re.sub(r'@(\w+)', ' [USER] ', t)
        t = re.sub(r'#(\w+)', r' \1 ', t)
        t = re.sub(r'(.)\1{2,}', r'\1\1', t)
        t = re.sub(r'[!]{2,}', ' [EXCITED] ', t)
        t = re.sub(r'[?]{2,}', ' [QUESTION] ', t)
        t = re.sub(r'[.]{3,}', ' [DOTS] ', t)
        t = re.sub(r'[^\w\s!?.,:;()]+', ' ', t)
        t = (t + " " + emojis).strip()
        t = re.sub(r'\d+', ' [NUMBER] ', t)
        t = re.sub(r'\s+', ' ', t).strip()

        T = " " + t + " "
        flags = {
            "ang": any(re.search(p, T) for p in self.ANGER) or self._has_emoji(T, self.EMO_ANGER),
            "sad": any(re.search(p, T) for p in self.SAD)   or self._has_emoji(T, self.EMO_SAD),
            "hap": any(re.search(p, T) for p in self.HAPPY) or self._has_emoji(T, self.EMO_HAPPY),
            "fer": any(re.search(p, T) for p in self.FEAR)  or self._has_emoji(T, self.EMO_FEAR),
        }
        cues = []
        if flags["ang"]: cues.append("angercue")
        if flags["sad"]: cues.append("sadcue")
        if flags["hap"]: cues.append("happycue")
        if flags["fer"]: cues.append("fearcue")
        if cues: t = t + " " + " ".join(cues)
        return t, flags

    # ============ Sentiment (calibrated) ============
    def analyze_sentiment(self, text: str) -> Dict:
        try:
            processed, flags = self._clean_and_cue(text)
            enc = self.tokenizer(
                processed, return_tensors="pt", truncation=True, padding=True,
                max_length=int(self.calib.get("max_length", 192))
            )
            with torch.no_grad():
                logits = self.model(**enc).logits.squeeze(0)

            # Kalibrasi logits sadness (index 2 pada default label)
            logits[2] += float(self.calib.get("sadness_bias", 0.0))
            if flags["ang"] and not flags["sad"]:
                logits[2] -= float(self.calib.get("delta_ang", 0.0))
            if flags["hap"] and not flags["sad"]:
                logits[2] -= float(self.calib.get("delta_hap", 0.0))
            if flags["fer"] and not flags["sad"]:
                logits[2] -= float(self.calib.get("delta_fear", 0.0))
            if flags["sad"]:
                logits[2] += float(self.calib.get("pos_sad", 0.0))

            probs = F.softmax(logits, dim=-1)
            confidence, idx = torch.max(probs, dim=0)

            if confidence.item() < self.confidence_threshold:
                return {'dominant_sentiment': 'neutral', 'confidence': confidence.item(), 'uncertain': True}

            return {
                'dominant_sentiment': self.labels[int(idx)],
                'confidence': confidence.item(),
                'uncertain': False
            }
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return {'dominant_sentiment': 'neutral', 'confidence': 0.5, 'uncertain': True}

    # ============ Style & empathy ============
    def analyze_user_style(self, user_input: str) -> Dict:
        style_profile = {}
        if re.search(r'\b(gue|gw)\b', user_input.lower()):
            style_profile['pronouns'] = 'gue'
        elif re.search(r'\b(aku|ak)\b', user_input.lower()):
            style_profile['pronouns'] = 'aku'
        elif re.search(r'\bsaya\b', user_input.lower()):
            style_profile['pronouns'] = 'saya'
        else:
            style_profile['pronouns'] = self.user_personality_profile['preferred_pronouns']

        exclamation_count = user_input.count('!')
        if exclamation_count >= 3: style_profile['exclamation_level'] = 'high'
        elif exclamation_count >= 1: style_profile['exclamation_level'] = 'medium'
        else: style_profile['exclamation_level'] = 'low'

        repeated_pattern = re.findall(r'(\w)\1{2,}', user_input.lower())
        style_profile['uses_repetition'] = len(repeated_pattern) > 0

        emoji_pattern = re.findall(r'[😀-🙏]', user_input)
        style_profile['uses_emoji'] = len(emoji_pattern) > 0

        slang_words = ['wkwk', 'anjir', 'sih', 'banget', 'dong', 'deh', 'kok', 'kayak', 'gimana']
        found_slang = [w for w in slang_words if w in user_input.lower()]
        style_profile['slang_words'] = found_slang
        style_profile['casualness'] = len(found_slang)

        word_count = len(user_input.split())
        if word_count <= 3: style_profile['message_length'] = 'short'
        elif word_count <= 10: style_profile['message_length'] = 'medium'
        else: style_profile['message_length'] = 'long'
        return style_profile

    def update_personality_profile(self, style_analysis: Dict):
        if style_analysis.get('pronouns'):
            self.user_personality_profile['preferred_pronouns'] = style_analysis['pronouns']
        if style_analysis.get('uses_emoji'):
            self.user_personality_profile['emoji_usage'] = True
        self.user_personality_profile['exclamation_tendency'] = style_analysis.get('exclamation_level', 'low')
        if self.turn_count > 3 and style_analysis.get('message_length') == 'long':
            self.user_personality_profile['openness_level'] = min(5, self.user_personality_profile['openness_level'] + 1)

    def create_natural_transition_comment(self, transition: str) -> str:
        transition_map = {
            'sadness → happy': random.choice([
                "Seneng deh denger kamu udah lebih baik!",
                "Asik banget! Kedengeran udah mulai membaik ya.",
                "Alhamdulillah ya udah mulai happy lagi."
            ]),
            'anger → happy': random.choice([
                "Asik banget! Seneng deh denger kamu udah lebih lega.",
                "Kedengeran udah nggak kesel lagi ya. Seneng banget!",
                "Syukur ya udah mulai happy lagi."
            ]),
            'anger → sadness': random.choice([
                "Hmm, kayaknya dari kesel jadi sedih ya.",
                "Sekarang lebih ke sedih gitu ya."
            ]),
            'fear → happy': random.choice([
                "Asik! Kedengeran udah nggak deg-degan lagi.",
                "Seneng deh udah nggak khawatir lagi.",
                "Alhamdulillah ya udah lega."
            ]),
            'sadness → anger': random.choice([
                "Hmm, dari sedih jadi kesel gitu ya.",
                "Sekarang malah jadi emosi ya."
            ]),
            'happy → sadness': random.choice([
                "Eh, kok jadi sedih lagi?",
                "Hmm, kayaknya mood-nya turun lagi ya."
            ]),
            'love → sadness': random.choice([
                "Hmm, dari bahagia jadi sedih ya.",
                "Kayaknya ada yang bikin berat lagi."
            ])
        }
        return transition_map.get(transition, "")

    def handle_special_cases(self, user_input: str, style_analysis: Dict) -> Optional[str]:
        user_lower = user_input.lower().strip()
        if user_lower in ['hmm', 'hm', 'mm']: return "Hmm, ada yang lagi kepikiran ya?"
        if user_lower in ['ok', 'oke', 'okay']: return "Oke. Ada yang mau diceritain lagi?"
        if user_lower in ['ya', 'iya', 'yup', 'yep']: return "Hmm, gimana perasaan kamu sekarang?"
        if re.match(r'^(wkwk|haha|hihi|hehe|lol|kwkw)+$', user_lower): return "Hahaha, ada yang lucu ya? Cerita dong!"
        if 'bingung' in user_lower:
            return "Kebingungan memang bikin nggak tenang. Mau coba diomongin pelan-pelan?" if style_analysis.get('pronouns') == 'saya' else "Hmm, bingung ya? Mau cerita apa yang bikin bingung?"
        love_confusion_patterns = [r'sayang.*tapi.*gak tau', r'cinta.*bingung', r'suka.*gimana']
        for p in love_confusion_patterns:
            if re.search(p, user_lower):
                return "Kedengeran sayang banget ya sama dia, tapi bingung juga harus gimana. Mau ngobrolin bareng-bareng?"
        return None

    def get_empathy_level(self) -> int:
        openness = self.user_personality_profile['openness_level']
        if self.turn_count <= 2: return 1
        elif self.turn_count <= 5 or openness <= 2: return 2
        else: return 3

    # --------- Style mirroring (termasuk 'kamu'→'lu' kalau gaya 'gue') ---------
    def mirror_user_style(self, base_response: str, style_analysis: Dict) -> str:
        response = base_response

        pronouns = style_analysis.get('pronouns', self.user_personality_profile['preferred_pronouns'])
        # first-person 'aku' → 'gue' / 'saya'
        if pronouns == 'gue':
            response = re.sub(r'\baku\b', 'gue', response)
            # second-person 'kamu' → 'lu'
            response = re.sub(r'\bkamu\b', 'lu', response)
            response = re.sub(r'\bKamu\b', 'Lu', response)
        elif pronouns == 'saya':
            response = re.sub(r'\baku\b', 'saya', response)

        ex_level = style_analysis.get('exclamation_level', 'low')
        if ex_level == 'high' and '.' in response: response = response.replace('.', '!')
        elif ex_level == 'medium' and response.count('!') == 0:
            if response.endswith('.'): response = response[:-1] + '!'

        if style_analysis.get('uses_repetition'):
            response = re.sub(r'\bbanget\b', 'bangettt', response)
            response = re.sub(r'\basik\b', 'asikk', response)

        if style_analysis.get('uses_emoji') and self.user_personality_profile['emoji_usage']:
            if not re.search(r'[😀-🙏]', response):
                emoji_map = {'happy':' 😊','sadness':' 😔','love':' 🥰','anger':' 😤','fear':' 😰','neutral':' 🙂'}
                response += emoji_map.get(getattr(self, 'current_sentiment', 'neutral'), '')
        return response

    def update_short_term_memory(self, user_input: str, bot_response: str, sentiment: str):
        self.short_term_memory.append({'user': user_input, 'bot': bot_response, 'sentiment': sentiment, 'turn': self.turn_count})
        if len(self.short_term_memory) > 3: self.short_term_memory.pop(0)

    def get_memory_context(self) -> str:
        if not self.short_term_memory: return ""
        parts = []
        for m in self.short_term_memory[-2:]:
            parts.append(f"User pernah bilang: '{m['user']}' (emosi: {m['sentiment']})")
        return "Konteks percakapan sebelumnya: " + " | ".join(parts) if parts else ""

    def detect_transition(self, current_sentiment: str) -> Optional[str]:
        if self.previous_sentiment and self.previous_sentiment != current_sentiment:
            return f"{self.previous_sentiment} → {current_sentiment}"
        return None

    def create_adaptive_prompt(self, user_input: str, current_sentiment: str,
                               transition: Optional[str], style_analysis: Dict) -> str:
        empathy_level = self.get_empathy_level()
        if current_sentiment in self.response_styles:
            style_options = self.response_styles[current_sentiment][f'level_{empathy_level}']
        else:
            style_options = self.response_styles['neutral'][f'level_{empathy_level}']
        base_style = random.choice(style_options)

        memory_context = self.get_memory_context()
        transition_comment = self.create_natural_transition_comment(transition) if transition else ""

        style_instructions = f"""
Gaya komunikasi user:
- Pronouns: {style_analysis.get('pronouns', 'aku')}
- Exclamation level: {style_analysis.get('exclamation_level', 'low')}
- Uses repetition: {style_analysis.get('uses_repetition', False)}
- Uses emoji: {style_analysis.get('uses_emoji', False)}
- Slang words: {', '.join(style_analysis.get('slang_words', []))}
- Message length: {style_analysis.get('message_length', 'medium')}
"""

        prompt = f"""
Kamu adalah chatbot empati yang sangat natural dan adaptif.

{memory_context}

User berkata: "{user_input}"
Emosi user: {current_sentiment} (confidence: medium)
Empathy level: {empathy_level}/3
Turn ke: {self.turn_count + 1}

{style_instructions}

Base response style: "{base_style}"

{f"Transition comment: {transition_comment}" if transition_comment else ""}

INSTRUKSI:
1. Respons harus natural, hindari format data/teknis
2. Mirror gaya komunikasi user (pronoun, exclamation, repetition)
3. Sesuaikan panjang respons dengan message user
4. Level empati: {"Simple validation" if empathy_level == 1 else "Deeper understanding" if empathy_level == 2 else "Meaningful connection"}
5. Jika ada transisi emosi, sebutkan secara halus dan natural
6. Maksimal 1–2 kalimat untuk turn awal, bisa lebih panjang seiring progression
7. Hindari bullet/daftar dan jangan gunakan baris baru; rangkum dalam 1 paragraf saja.
"""
        return prompt

    # ---------- FLATTENER (hapus newline, bullet, markdown) ----------
    def _flatten(self, s: str) -> str:
        if not isinstance(s, str): s = str(s)
        s = s.replace("\r", " ").replace("\n", " ")
        s = s.replace("**", "").replace("*", "").replace("•", "").replace("—", "-")
        s = re.sub(r'\s+', ' ', s)
        s = re.sub(r'\s+([,.;:!?])', r'\1', s)
        s = re.sub(r'(Saran ringan:)\s*(Saran ringan:)+', r'\1 ', s, flags=re.I)
        s = re.sub(r'(Rekomendasi ringan:)\s*(Rekomendasi ringan:)+', r'\1 ', s, flags=re.I)
        return s.strip()

    # ---------- Preface (izin/bridging) ----------
    def _preface_reco(self, sentiment: str, asked: bool, empathy_level: int) -> str:
        soft = {
            'sadness': "Aku kebayang ini berat.",
            'fear': "Deg-degan kayak gini memang nggak enak.",
            'anger': "Pantes sih kalau kesel.",
            'happy': "Seneng denger kabar baikmu.",
            'love': "Kedengeran sayang banget ya.",
            'neutral': "Biar bantuannya makin pas."
        }
        opener = soft.get(sentiment, soft['neutral'])
        ask = "Kalau kamu mau, aku ada saran kecil" if not asked else "Oke, aku share saran kecil"
        tone = " yang biasanya membantu" if empathy_level >= 2 else ""
        return f"{opener} {ask}{tone}: "

    def _preface_hotline(self) -> str:
        return (" Aku khawatir sama keselamatan kamu; kalau kamu merasa tidak aman sekarang, "
                "prioritasnya keselamatan ya.")

    # ---------- Rekomendasi & Krisis helpers ----------
    def _explicit_reco_requested(self, text: str) -> bool:
        t = text.lower()
        return any(re.search(p, t) for p in self.EXPLICIT_RECO_WORDS)

    def _detect_crisis(self, text: str) -> Dict:
        t = text.lower()
        is_self_harm = any(re.search(p, t) for p in self.SELF_HARM)
        is_harm_others = any(re.search(p, t) for p in self.HARM_OTHERS)
        is_abuse = any(re.search(p, t) for p in self.ABUSE)
        is_crisis = is_self_harm or is_harm_others or is_abuse
        crisis_type = 'self_harm' if is_self_harm else ('harm_others' if is_harm_others else ('abuse' if is_abuse else None))
        return {'is_crisis': is_crisis, 'type': crisis_type}

    def _negative_streak(self, k: int = 2) -> bool:
        if len(self.short_term_memory) < k: return False
        neg = {'sadness','fear','anger'}
        last = [m['sentiment'] for m in self.short_term_memory[-k:]]
        return all(s in neg for s in last)

    def _should_offer_reco(self, user_input: str, sentiment: str) -> bool:
        if self._explicit_reco_requested(user_input): return True
        if self.turn_count < 2: return False
        if self.turn_count - self.last_reco_turn < self.reco_cooldown: return False
        if sentiment in {'sadness','fear','anger'} or self._negative_streak(2):
            return True
        return False

    def _build_recommendation_inline(self, sentiment: str, asked: bool, empathy_level: int) -> str:
        bank = {
            'fear': [
                "coba napas 4-4-4 satu–dua menit",
                "tulis satu langkah kecil yang bisa kamu ambil sekarang"
            ],
            'sadness': [
                "lakukan aktivitas mini 5 menit seperti mandi hangat atau jalan ringan",
                "tulis satu hal yang bikin kamu tetap bertahan hari ini"
            ],
            'anger': [
                "ambil jeda 90 detik untuk nenangin tubuh",
                "bedakan yang bisa kamu kontrol dan yang tidak"
            ],
            'love': [
                "kalau nyaman, pakai format 'aku merasa..., karena...'",
                "jaga batas sehat dengan komunikasi jujur"
            ],
            'happy': [
                "rayakan momen kecil dan simpan catatannya",
                "bagikan kabar baik ke orang yang kamu percaya"
            ],
            'neutral': [
                "kalau mau, ceritakan sedikit detail biar aku bisa bantu lebih pas",
                "tarik napas pelan beberapa kali sambil minum air"
            ]
        }
        tips = bank.get(sentiment, bank['neutral'])
        preface = self._preface_reco(sentiment, asked, empathy_level)
        body = "; ".join(tips) + "."
        return preface + body

    def _build_hotline_inline(self) -> str:
        return (self._preface_hotline() + " " + self.HOTLINES_ID + " Kamu aman sekarang?")

    # ---------------- MAIN CHAT ----------------
    def chat(self, user_input: str) -> Dict:
        try:
            style_analysis = self.analyze_user_style(user_input)

            # Sentiment first
            sentiment_result = self.analyze_sentiment(user_input)
            current_sentiment = sentiment_result['dominant_sentiment']
            self.current_sentiment = current_sentiment

            # Update profil & transisi
            self.update_personality_profile(style_analysis)
            transition = self.detect_transition(current_sentiment)

            # Prompt ke LLM
            adaptive_prompt = self.create_adaptive_prompt(user_input, current_sentiment, transition, style_analysis)
            response = self.chat_session.send_message(adaptive_prompt)
            response_text = response.text.strip()
            mirrored_response = self.mirror_user_style(response_text, style_analysis)

            # Krisis & rekomendasi (inline)
            crisis = self._detect_crisis(user_input)
            asked  = self._explicit_reco_requested(user_input)
            if crisis['is_crisis']:
                base = mirrored_response.split(".")[0] if "." in mirrored_response else mirrored_response
                mirrored_response = self.mirror_user_style(base + "." + self._build_hotline_inline(), style_analysis)
            else:
                offer = self._should_offer_reco(user_input, current_sentiment)
                if offer or asked:
                    reco_inline = self._build_recommendation_inline(current_sentiment, asked, self.get_empathy_level())
                    mirrored_response = self.mirror_user_style(mirrored_response + " " + reco_inline, style_analysis)
                    self.last_reco_turn = self.turn_count

            # Paksa satu paragraf bersih
            mirrored_response = self._flatten(mirrored_response)

            # Memory & state
            self.update_short_term_memory(user_input, mirrored_response, current_sentiment)
            self.previous_sentiment = current_sentiment
            self.turn_count += 1

            return {
                'response': mirrored_response,
                'sentiment': current_sentiment,
                'confidence': sentiment_result.get('confidence', 1.0),
                'transition': transition,
                'empathy_level': self.get_empathy_level(),
                'style_analysis': style_analysis,
                'special_case': False
            }
        except Exception as e:
            print(f"Chat error: {e}")
            return {
                'response': "Maaf, lagi ada gangguan. Coba lagi ya.",
                'sentiment': 'neutral',
                'confidence': 0.5,
                'transition': None,
                'empathy_level': 1,
                'style_analysis': {},
                'special_case': False
            }

    def reset_chat_session(self):
        try:
            self.chat_session = self.chat_model.start_chat(history=[])
            self.turn_count = 0
            self.short_term_memory = []
            self.previous_sentiment = None
            self.user_personality_profile = {
                'preferred_pronouns': 'aku',
                'formality_level': 'casual',
                'emoji_usage': False,
                'exclamation_tendency': 'low',
                'openness_level': 1
            }
            self.last_reco_turn = -10
            print("🔄 Session reset")
        except Exception as e:
            print(f"Reset error: {e}")
