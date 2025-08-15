# chatbot/rasa_chatbot.py
import os, json, re, random, torch
import torch.nn.functional as F
from typing import Dict, Optional, List

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import google.generativeai as genai


class RasaChatbot:
    def __init__(self, model_name: str, gemini_api_key: str):
        print("v2 Initializing RasaChatbot (calibrated sentiment)…")
        self.model_name = model_name
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN", None)

        # 1) Load IndoBERT model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=hf_token)
        self.model.eval()

        # 2) Load calibration (lokal > HF > default)
        calib_path = os.getenv("CALIB_JSON")
        self.calib = None
        if calib_path and os.path.exists(calib_path):
            with open(calib_path, "r") as f:
                self.calib = json.load(f)
        else:
            fname = os.getenv("CALIB_JSON_HF", "calibration.json")
            try:
                downloaded = hf_hub_download(repo_id=model_name, repo_type="model", filename=fname, use_auth_token=hf_token)
                with open(downloaded, "r") as f:
                    self.calib = json.load(f)
            except Exception as e:
                print(f"calibration.json not found on Hub ({fname}). Using defaults. Error: {e}")
                self.calib = {
                    "labels": ["anger","happy","sadness","love","fear"],
                    "sadness_bias": 0.0, "delta_ang": 0.0, "delta_hap": 0.0, "delta_fear": 0.0, "pos_sad": 0.0,
                    "max_length": 192
                }

        # 3) Label order mengikuti kalibrasi (WAJIB konsisten dengan training)
        self.labels = self.calib.get("labels", ["anger","happy","sadness","love","fear"])

        # 4) Gemini
        genai.configure(api_key=gemini_api_key)
        self.chat_model = genai.GenerativeModel("gemini-2.0-flash")
        self.chat_session = self.chat_model.start_chat(history=[])

        # 5) Cue keywords & emoji (ringkas, selaras training)
        self.ANGER = [r'\bmarah\b', r'\bkesal\b', r'\bkesel\b', r'\bjengkel\b', r'\bdongkol\b', r'\bsebal\b', r'\bsebel\b', r'\bgeram\b', r'\bbenci\b', r'\bmurka\b', r'\bemosi\b', r'\bngamuk\b']
        self.SAD   = [r'\bsedih\b', r'\bkecewa\b', r'\bterpuruk\b', r'\bgalau\b', r'\bmurung\b', r'\bhancur\b', r'\bpatah hati\b', r'\bdown\b', r'\bmenangis\b', r'\bnangis\b', r'\bduka(cita)?\b', r'\bterluka\b', r'\bsunyi\b', r'\bkesepian\b', r'\bsepi\b', r'\bputus asa\b', r'\bhampa\b']
        self.HAPPY = [r'\bbahagia\b', r'\bsenang\b', r'\bgembira\b', r'\bsuka\b', r'\bceria\b', r'\blega\b', r'\bpuas\b', r'\bsemangat\b', r'\bbersyukur\b']
        self.FEAR  = [r'\btakut\b', r'\bcemas\b', r'\bkhawatir\b', r'\bwas[- ]?was\b', r'\bdeg(-| )?degan\b', r'\bparno\b', r'\bparanoid\b', r'\bgentar\b', r'\bketakutan\b', r'\bngeri\b']

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

        # >>> ADDED: state untuk rekomendasi & krisis
        self.last_reco_turn = -99           # turn terakhir rekomendasi muncul
        self.neg_streak = 0                 # jumlah turn negatif berturut-turut
        self.reco_cooldown = 1              # jeda minimal antar rekomendasi
        self.current_sentiment = "neutral"  # cache emosi berjalan
        
        # 7) Response styles (tetap dari kode lama)
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

        # >>> ADDED: bank saran ringan berbasis psikologi (aman & non-diagnostik)
        self._reco_bank = {
            "sadness": [
                "Coba napas 4-7-8 selama ±2 menit untuk menurunkan ketegangan.",
                "Tulis 3 pikiran yang mengganggu → buat 1 kalimat reframing yang lebih realistis."
            ],
            "fear": [
                "Lakukan grounding 5-4-3-2-1 (lihat 5 benda, raba 4, dengar 3, cium 2, rasakan 1).",
                "Coba muscle relaxation singkat: tegang-kan lalu rileks-kan otot dari bahu ke kaki."
            ],
            "anger": [
                "Ambil time-out 90 detik sebelum merespons; fokus ke napas agar amygdala tenang.",
                "Tulis uneg-uneg tanpa kirim (venting aman), baru putuskan langkah berikutnya."
            ],
            "happy": [
                "Pertahankan mood dengan gratitude 3 hal hari ini."
            ],
            "love": [
                "Coba komunikasi asertif 2 kalimat: ‘Aku merasa… ketika… Aku butuh…’"
            ],
            "neutral": [
                "Kalau mau, kita coba labeling emosi: kasih nama perasaan yang paling dekat sekarang."
            ]
        }
        
        print("RasaChatbot ready.")

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
        
    # >>> ADDED: helper deteksi krisis & format rekomendasi
    def _is_crisis(self, text: str) -> bool:
        t = (text or "").lower()
        crisis_kw = [
            "bunuh diri","akhiri hidup","mengakhiri hidup","gak mau hidup","tidak mau hidup",
            "self harm","melukai diri","nyakitin diri","menyakiti diri","putus asa banget"
        ]
        return any(k in t for k in crisis_kw)

    def _cooldown_ok(self) -> bool:
        return (self.turn_count - self.last_reco_turn) > self.reco_cooldown

    def _get_recommendations(self, emo: str, max_items: int = 2) -> List[str]:
        return self._reco_bank.get(emo, self._reco_bank["neutral"])[:max_items]

    def _should_offer_counselor(self, emo: str) -> bool:
        neg = {"sadness","fear","anger"}
        if emo in neg: self.neg_streak += 1
        else: self.neg_streak = 0
        return self.neg_streak >= 2

    def _format_block(self, title: str, bullets: List[str]) -> str:
        if not bullets: return ""
        lines = [f"**{title}:**"]
        for b in bullets:
            lines.append(f"• {b}")
        return "\n".join(lines)

    # ============ Sentiment (calibrated) ============
    def analyze_sentiment(self, text: str) -> Dict:
        """Calibrated sentiment with confidence gate"""
        try:
            processed, flags = self._clean_and_cue(text)
            enc = self.tokenizer(
                processed, return_tensors="pt", truncation=True, padding=True,
                max_length=int(self.calib.get("max_length", 192))
            )
            with torch.no_grad():
                logits = self.model(**enc).logits.squeeze(0)

            # Kalibrasi logit sadness (index 2)
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

    def mirror_user_style(self, base_response: str, style_analysis: Dict) -> str:
        response = base_response
        pronouns = style_analysis.get('pronouns', self.user_personality_profile['preferred_pronouns'])
        if pronouns == 'gue': response = response.replace('aku', 'gue')
        elif pronouns == 'saya': response = response.replace('aku', 'saya')

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

    def create_adaptive_prompt(self, user_input: str, current_sentiment: str, transition: Optional[str], style_analysis: Dict) -> str:
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
1. Respons harus natural, hindari format data atau teknis
2. Mirror gaya komunikasi user (pronoun, exclamation, repetition pattern)
3. Sesuaikan panjang respons dengan message user
4. Level empati: {"Simple validation" if empathy_level == 1 else "Deeper understanding" if empathy_level == 2 else "Meaningful connection"}
5. Jika ada transisi emosi, sebutkan secara halus dan natural
6. Maksimal 1-2 kalimat untuk turn awal, bisa lebih panjang seiring progression
7. Gunakan slang yang sama dengan user jika ada
"""
        # >>> ADDED: kebijakan rekomendasi/krisis supaya LLM ikut format yang sama
        crisis_flag = self._is_crisis(user_input)
        policy = f"""
Kebijakan Rekomendasi (WAJIB DIIKUTI):
- Jika CRISIS=true: JANGAN beri tips biasa. Fokus dukungan singkat + berikan hotline Indonesia:
  • SEJIWA 119 ext. 8 (dukungan psikologis awal)
  • Halo Kemenkes 1500-567 (informasi/pengaduan kesehatan)
  • SAPA 129 / WA 08111-129-129 (kekerasan perempuan & anak)
  • Gawat darurat medis: PSC 119.
- Jika CRISIS=false dan emosi ∈ {{sadness,fear,anger}} dengan confidence≥0.65:
  Tambahkan blok 'Saran ringan' (maks 2 poin) berbasis teknik psikologi yang aman (napas 4-7-8, grounding 5-4-3-2-1, reframing kognitif, muscle relaxation).
- Jika emosi ∈ {{happy,neutral}}: jangan beri saran kecuali diminta, atau ada transisi negatif→positif (beri 1 saran maintain mood).
- Jika 2 turn berturut-turut emosi negatif ATAU user minta bantuan profesional:
  Akhiri dengan 1 kalimat ajakan: “Mau aku hubungkan ke konselor di aplikasi ini?”
Format keluaran:
1) Paragraf respons empatik (1–3 kalimat).
2) (Opsional) 'Saran ringan:' bullet pendek.
3) (Opsional) Ajakan ke konselor.
4) (Krisis) 'Hotline:' daftar ringkas di atas.
CRISIS={str(crisis_flag).lower()}
"""
        prompt = f"{prompt}\n{policy}"
        return prompt

    def chat(self, user_input: str) -> Dict:
        try:
            style_analysis = self.analyze_user_style(user_input)
            special = self.handle_special_cases(user_input, style_analysis)
            if special:
                mirrored = self.mirror_user_style(special, style_analysis)
                self.turn_count += 1
                return {
                    'response': mirrored, 'sentiment': 'neutral', 'confidence': 1.0,
                    'transition': None, 'special_case': True, 'style_analysis': style_analysis,
                    'empathy_level': self.get_empathy_level()
                }

            sentiment_result = self.analyze_sentiment(user_input)
            current_sentiment = sentiment_result['dominant_sentiment']
            self.current_sentiment = current_sentiment

            self.update_personality_profile(style_analysis)
            transition = self.detect_transition(current_sentiment)
            adaptive_prompt = self.create_adaptive_prompt(user_input, current_sentiment, transition, style_analysis)

            response = self.chat_session.send_message(adaptive_prompt)
            response_text = response.text.strip()
            mirrored_response = self.mirror_user_style(response_text, style_analysis)

            # >>> ADDED: sisipkan blok rekomendasi/krisis/CTA konselor langsung ke response
            confidence = float(sentiment_result.get('confidence', 1.0))
            is_crisis = self._is_crisis(user_input)
            blocks: List[str] = []

            if is_crisis:
                # Krisis: hotline nasional
                hotline = [
                    "SEJIWA 119 ext. 8 (dukungan psikologis awal nasional)",
                    "Halo Kemenkes 1500-567 (informasi/pengaduan kesehatan)",
                    "SAPA 129 / WA 08111-129-129 (kekerasan perempuan & anak)",
                    "Keadaan gawat darurat medis: hubungi PSC 119"
                ]
                blocks.append(self._format_block("Hotline", hotline))
            else:
                # Rekomendasi ringan hanya untuk emosi negatif + confidence cukup + cooldown
                if current_sentiment in {"sadness","fear","anger"} and confidence >= 0.65 and self._cooldown_ok():
                    recos = self._get_recommendations(current_sentiment, max_items=2)
                    blocks.append(self._format_block("Saran ringan", recos))
                    self.last_reco_turn = self.turn_count

                # Transisi negatif -> positif: beri 1 saran maintain mood
                if transition and transition.startswith(("sadness", "fear", "anger")) and current_sentiment in {"happy","neutral"}:
                    blocks.append(self._format_block("Lanjutkan hal baik", self._get_recommendations("happy", 1)))

                # Ajakan ke konselor jika negatif beruntun
                if self._should_offer_counselor(current_sentiment):
                    blocks.append("**Butuh bantuan profesional?** Mau aku hubungkan ke **konselor** di aplikasi ini?")

            # Susun final response (dalam satu string yang sama)
            extra = ("\n\n" + "\n\n".join([b for b in blocks if b])).strip() if blocks else ""
            final_text = mirrored_response + (("\n\n" + extra) if extra else "")

            # Update memory & counters
            self.update_short_term_memory(user_input, final_text, current_sentiment)
            self.previous_sentiment = current_sentiment
            self.turn_count += 1

            return {
                'response': final_text,                                # >>> MODIFIED (pakai final_text)
                'sentiment': current_sentiment,
                'confidence': confidence,
                'transition': transition,
                'empathy_level': self.get_empathy_level(),
                'style_analysis': style_analysis,
                'special_case': False
            }
        except Exception as e:
            print(f"Chat error: {e}")
            return {'response': "Maaf, lagi ada gangguan. Coba lagi ya.", 'sentiment': 'neutral',
                    'confidence': 0.5, 'transition': None, 'empathy_level': 1,
                    'style_analysis': {}, 'special_case': False}

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
            # >>> ADDED: reset state rekomendasi
            self.last_reco_turn = -99
            self.neg_streak = 0
            self.current_sentiment = "neutral"
            print("Session reset")
        except Exception as e:
            print(f"Reset error: {e}")


