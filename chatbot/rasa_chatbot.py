# chatbot/rasa_chatbot.py
import os, json, re, random, torch
import torch.nn.functional as F
from typing import Dict, Optional, List

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import google.generativeai as genai


class RasaChatbot:
    def __init__(self, model_name: str, gemini_api_key: str):
        print("v2.1 Initializing RasaChatbot (calibrated sentiment, humanized)…")
        self.model_name = model_name
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN", None)

        # 1) Load IndoBERT model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=hf_token)
        self.model.eval()

        # 2) Load calibration (local > HF > default)
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
                print(f"⚠️ calibration.json not found on Hub ({fname}). Using defaults. Error: {e}")
                self.calib = {
                    "labels": ["anger","happy","sadness","love","fear"],
                    "sadness_bias": 0.0, "delta_ang": 0.0, "delta_hap": 0.0, "delta_fear": 0.0, "pos_sad": 0.0,
                    "max_length": 192
                }

        # 3) Label order mengikuti kalibrasi
        self.labels = self.calib.get("labels", ["anger","happy","sadness","love","fear"])

        # 4) Gemini
        genai.configure(api_key=gemini_api_key)
        self.chat_model = genai.GenerativeModel("gemini-2.0-flash")
        self.chat_session = self.chat_model.start_chat(history=[])

        # 5) Cue keywords & emoji
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

        # 7) Response styles (dipakai sebagai “rasa”, tapi nanti di-recipe dan di-randomize)
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

        # 8) Krisis: frasa berisiko (non-diagnostik, untuk rujukan saja)
        self.CRISIS_PATTERNS = [
            r'\b(pengen|ingin|mau)\s*(mati|mengakhiri|ngakhirin)\b',
            r'\b(gak|tidak)\s*ada\s*harapan\b',
            r'\bnyakitin\s*diri\b|\bself[- ]?harm\b',
            r'\bbunuh\s*diri\b|\bsuicide\b',
            r'\baku\s*berbahaya\s*buat\s*(diri|orang)\b',
        ]

        print("✅ RasaChatbot ready.")

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
                "Seneng denger kamu udah lebih lega.",
                "Asik, pelan-pelan ngerasa lebih baik ya."
            ]),
            'anger → happy': random.choice([
                "Syukur bisa lebih adem sekarang.",
                "Asik, rasanya udah nggak seketat tadi ya."
            ]),
            'anger → sadness': random.choice([
                "Dari kesel jadi kerasa sedih, ya.",
                "Kayaknya sekarang beratnya lebih ke sedih."
            ]),
            'fear → happy': random.choice([
                "Lega ya bisa bernafas lebih tenang.",
                "Asik, deg-degannya mulai turun."
            ]),
            'sadness → anger': random.choice([
                "Kadang dari sedih bisa jadi kesal juga, wajar kok.",
            ]),
            'happy → sadness': random.choice([
                "Eh, mood-nya agak turun ya.",
            ]),
            'love → sadness': random.choice([
                "Perasaan hangatnya sempat ada, terus jadi berat lagi ya.",
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

    # ======== Humanizing components (baru) ========
    def _maybe_micro_story(self, sentiment: str) -> Optional[str]:
        """
        Cerita pendek (1-2 kalimat), kadang-kadang saja supaya terasa manusiawi.
        Frekuensi kecil saat awal (turn<=2) dan meningkat sedikit setelahnya.
        """
        chance = 0.10 if self.turn_count <= 2 else 0.20
        if random.random() > chance: 
            return None
        bank = {
            'sadness': [
                "Dulu aku pernah ngerasain hari-hari yang rasanya berat banget juga; waktu itu aku mulai dari hal kecil—rapihin tempat tidur tiap pagi—aneh tapi bikin ada rasa kendali sedikit.",
                "Aku pernah ngelewatin masa yang bikin lelah batin; pelan-pelan, jalan sore tanpa headset malah bantu aku dengerin diri sendiri."
            ],
            'anger': [
                "Aku juga pernah kebawa emosi sama hal sepele; nafas 4-4-6 waktu itu nolong aku buat nggak meledak.",
                "Pernah banget aku kesel sama orang deket; nulis draft pesan dulu di notes bikin aku nggak nyesel setelahnya."
            ],
            'fear': [
                "Aku pernah ngerasa deg-degan tanpa alasan jelas; waktu itu aku ngecek realita dengan nulis tiga hal yang benar-benar terjadi saat itu.",
                "Rasa was-was itu aku kenal; ngitung pelan 5 benda yang kelihatan di sekitar lumayan nurunin tegang."
            ],
            'happy': [
                "Momen kecil yang bikin seneng itu suka datang pas nggak diduga; nyimpen di catatan syukur bikin aku inget pas lagi turun.",
                "Asik ya, rasa lega gini sering jadi titik mulai buat langkah kecil berikutnya."
            ],
            'love': [
                "Perasaan hangat ke seseorang pernah bikin aku berani jujur pelan-pelan; ternyata ritme yang nyaman buat dua pihak itu kunci.",
                "Aku pernah ngerasa deg-degan manis; nulis tiga hal yang aku apresiasi bikin perasaan makin jelas."
            ],
            'neutral': [
                "Kadang momen netral justru enak buat nyusun langkah; aku suka mulai dari pertanyaan kecil: ‘Satu hal yang mau aku rasakan dari hari ini apa?’",
                "Ada hari-hari yang datar; biasanya aku pilih satu tugas 5 menit biar mesin nyala dulu."
            ]
        }
        return random.choice(bank.get(sentiment, bank['neutral']))

    def _safe_recommendations(self, sentiment: str) -> str:
        """
        Rekomendasi ringan, aman, dan opsional. 1-2 butir.
        """
        base = {
            'sadness': [
                "Coba ‘grounding 5-4-3-2-1’ sebentar untuk nyambung ke sekitar.",
                "Kalau kuat, keluar sebentar cari cahaya/udara, terus minum air hangat."
            ],
            'anger': [
                "Tarik nafas 4 detik, tahan 4, hembus 6; ulang 5 kali.",
                "Tunda balas chat/telepon dengan draft; kirim setelah 30 menit."
            ],
            'fear': [
                "Coba ucapkan pelan 3 hal yang *benar-benar* kamu lihat/dengar/rasakan saat ini.",
                "Batasi konsumsi berita/scroll 15 menit, lalu evaluasi perasaan lagi."
            ],
            'happy': [
                "Catat satu hal yang bikin kamu bangga hari ini.",
                "Simpan momen ini; kirim ‘terima kasih’ ke diri sendiri."
            ],
            'love': [
                "Kalau mau, tulis pesan simpel yang jujur dan ringan.",
                "Jaga ritme: bagikan cerita secukupnya, dengar balik responnya."
            ],
            'neutral': [
                "Pilih tugas 5 menit (mis. rapikan meja kecil).",
                "Minum air, tarik nafas dalam 3 kali, lalu cek ulang prioritas."
            ]
        }
        picks = random.sample(base.get(sentiment, base['neutral']), k=1)
        return "Saran ringan: " + "; ".join(picks)

    def _should_attach_reco(self, sentiment: str) -> bool:
        """
        Tidak selalu memberi rekomendasi; peluang naik kalau user mengeluh/negatif.
        """
        base_chance = {
            'sadness': 0.6, 'anger': 0.6, 'fear': 0.6,
            'neutral': 0.3, 'love': 0.25, 'happy': 0.25
        }
        # Awal percakapan jangan terlalu ‘mengarahkan’
        if self.turn_count <= 1:
            return random.random() < (base_chance.get(sentiment, 0.3) * 0.5)
        return random.random() < base_chance.get(sentiment, 0.3)

    def _gentle_distance_phrase(self) -> str:
        """
        Agar tidak terasa ‘terlalu dekat’ di awal.
        """
        choices = [
            "Kalau kamu berkenan, ceritain pelan-pelan aja.",
            "Aku dengerin dari sini, nggak buru-buru kok.",
            "Boleh sepatah dua patah dulu, senyaman kamu."
        ]
        # Lebih banyak ‘jarak’ di 2 turn pertama
        return random.choice(choices if self.turn_count <= 2 else choices + [
            "Ambil tempo yang nyaman buatmu, aku ngikutin."
        ])

    # ======== Krisis detection & referral (lembut) ========
    def _detect_crisis(self, text: str) -> bool:
        t = text.lower()
        return any(re.search(p, t) for p in self.CRISIS_PATTERNS)

    def _crisis_referral(self, style_analysis: Dict) -> str:
        """
        Rujukan lembut saat krisis. Tidak menggurui, langsung to-the-point tapi hangat.
        (Tanpa nomor spesifik — silakan sesuaikan di layer UI/produk jika perlu.)
        """
        pron = style_analysis.get('pronouns', 'aku')
        # Kalimat singkat, non-judgmental:
        lines = [
            "Aku khawatir sama keselamatanmu.",
            "Kalau kamu merasa berisiko melukai diri/ada bahaya sekarang, minta bantuan langsung ya.",
            "Hubungi layanan darurat setempat, tenaga kesehatan jiwa, atau orang tepercaya di dekatmu.",
        ]
        # Variasi gaya:
        if pron == 'saya':
            lines[0] = "Saya khawatir dengan keselamatan Anda."
        return " ".join(lines)

    # ======== Prompt builder ========
    def create_adaptive_prompt(self, user_input: str, current_sentiment: str, transition: Optional[str], style_analysis: Dict) -> str:
        empathy_level = self.get_empathy_level()
        if current_sentiment in self.response_styles:
            style_options = self.response_styles[current_sentiment][f'level_{empathy_level}']
        else:
            style_options = self.response_styles['neutral'][f'level_{empathy_level}']
        base_style = random.choice(style_options)

        memory_context = self.get_memory_context()
        transition_comment = self.create_natural_transition_comment(transition) if transition else ""

        # ——— Resep respons (diacak) ———
        recipes = [
            # 1) Validasi singkat + ajakan pelan
            "Mulai dengan validasi hangat 1 kalimat, lanjutkan 1 pertanyaan terbuka yang ringan.",
            # 2) Parafrase + refleksi perasaan
            "Parafrase inti cerita user dengan kata berbeda, lalu sebutkan perasaan yang mungkin dirasakan.",
            # 3) Pikiran & tubuh
            "Tunjukkan kamu memperhatikan sinyal tubuh/energi (mis. lelah/tegang) dan beri ruang.",
            # 4) Presence minimal
            "Jawab pendek tapi hangat, tanpa memaksa user untuk menjelaskan panjang.",
            # 5) Problem–free talk
            "Ajak bahas satu momen kecil yang terasa aman/neutral sebelum menyentuh masalah inti.",
        ]
        picked_recipe = random.choice(recipes)

        # Instruksi gaya & guardrails ke LLM
        style_instructions = f"""
GAYA USER:
- Pronoun: {style_analysis.get('pronouns', 'aku')}
- Tanda seru: {style_analysis.get('exclamation_level', 'low')}
- Repetisi: {style_analysis.get('uses_repetition', False)}
- Emoji: {style_analysis.get('uses_emoji', False)}
- Slang: {', '.join(style_analysis.get('slang_words', [])) or '-'}
- Panjang pesan user: {style_analysis.get('message_length', 'medium')}

RESEP RESPON (acak):
- {picked_recipe}

ATURAN PANJANG:
- Jika user pendek → 1 kalimat (maks 16 kata).
- Jika user sedang → 1–2 kalimat.
- Jika user panjang → 2–4 kalimat.
- Hindari bullet/angka; jangan terlihat seperti sistem.

VARIASI:
- Jangan ulang pola/frasa persis dari respons sebelumnya.
- Jika sesuai, sisipkan 1 metafora ringan atau peribahasa singkat (opsional).
- Hindari “kayaknya lagi susah banget ya” berulang; gunakan padanan alami lain.

REKOMENDASI:
- Rekomendasi hanya jika bermanfaat & aman; maksimal 1 kalimat, tanpa menggurui.
- Jangan berikan saran medis/diagnosis. Hindari menyebut obat atau tindakan berisiko.
"""

        # Catatan: micro-story & rekomendasi akan disuntik setelah model merespons.
        prompt = f"""
Kamu adalah pendengar yang empatik dan terasa manusiawi (tidak kaku).
Tugasmu menanggapi pesan user secara natural dan hangat.

{memory_context}

User: "{user_input}"
Emosi dominan (dari model): {current_sentiment}
Empathy level: {empathy_level}/3
Turn ke: {self.turn_count + 1}

Base style cue: "{base_style}"
{f"Catat perubahan emosi: {transition_comment}" if transition_comment else ""}

{style_instructions}

OUTPUT:
- Tulis respons final saja (tanpa label, tanpa daftar).
- Gunakan gaya bahasa yang selaras dengan user.
- Jaga jarak sosial yang sehat di awal; gunakan kalimat seperti: "{self._gentle_distance_phrase()}"
"""
        return prompt

    # ======== Memory helpers ========
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

    # ======== Style mirroring ========
    def mirror_user_style(self, base_response: str, style_analysis: Dict) -> str:
        response = base_response.strip()

        # Pronoun
        pronouns = style_analysis.get('pronouns', self.user_personality_profile['preferred_pronouns'])
        if pronouns == 'gue':
            response = response.replace('aku', 'gue').replace('saya', 'gue')
        elif pronouns == 'saya':
            response = response.replace('aku', 'saya')

        # Tanda seru
        ex_level = style_analysis.get('exclamation_level', 'low')
        if ex_level == 'high' and '.' in response: 
            response = response.replace('.', '!')
        elif ex_level == 'medium' and '!' not in response and response.endswith('.'):
            response = response[:-1] + '!'

        # Repetisi
        if style_analysis.get('uses_repetition'):
            response = re.sub(r'\bbanget\b', 'bangettt', response)
            response = re.sub(r'\basik\b', 'asikk', response)

        # Emoji (opsional, mengikuti user)
        if style_analysis.get('uses_emoji') and self.user_personality_profile['emoji_usage']:
            if not re.search(r'[😀-🙏]', response):
                emoji_map = {'happy':' 😊','sadness':' 😔','love':' 🥰','anger':' 😤','fear':' 😰','neutral':' 🙂'}
                response += emoji_map.get(getattr(self, 'current_sentiment', 'neutral'), '')

        # Panjang respons menyesuaikan panjang user
        msg_len = style_analysis.get('message_length', 'medium')
        if msg_len == 'short':
            # Pangkas ke ±1 kalimat
            response = re.split(r'(?<=[.!?])\s+', response)[0]

        return response

    # ======== Chat utama (JSON tidak diubah) ========
    def chat(self, user_input: str) -> Dict:
        try:
            style_analysis = self.analyze_user_style(user_input)

            # Special cases satu-liner
            special = self.handle_special_cases(user_input, style_analysis)
            if special:
                mirrored = self.mirror_user_style(special, style_analysis)
                self.turn_count += 1
                return {
                    'response': mirrored, 'sentiment': 'neutral', 'confidence': 1.0,
                    'transition': None, 'special_case': True, 'style_analysis': style_analysis,
                    'empathy_level': self.get_empathy_level()
                }

            # 1) Sentiment wajib lewat model
            sentiment_result = self.analyze_sentiment(user_input)
            current_sentiment = sentiment_result['dominant_sentiment']
            self.current_sentiment = current_sentiment

            # 2) Update profil gaya
            self.update_personality_profile(style_analysis)

            # 3) Transisi emosi
            transition = self.detect_transition(current_sentiment)

            # 4) Krisis check (hanya rujukan lembut saat terdeteksi)
            crisis_text = None
            if self._detect_crisis(user_input):
                crisis_text = self._crisis_referral(style_analysis)

            # 5) Prompt adaptif ke LLM (tanpa memasukkan saran/cerita dulu)
            adaptive_prompt = self.create_adaptive_prompt(user_input, current_sentiment, transition, style_analysis)
            response = self.chat_session.send_message(adaptive_prompt)
            response_text = (response.text or "").strip()

            # 6) Sisipkan micro-story jika cocok
            micro = self._maybe_micro_story(current_sentiment)
            if micro:
                # Tempel sebagai kalimat terakhir (maks 1-2 kalimat total kalau user pendek)
                if style_analysis.get('message_length') == 'short':
                    response_text = re.split(r'(?<=[.!?])\s+', response_text)[0]
                response_text = response_text + (" " if response_text and not response_text.endswith(('.', '!', '?')) else " ") + micro

            # 7) Tambahkan rekomendasi aman bila perlu
            if self._should_attach_reco(current_sentiment):
                reco = self._safe_recommendations(current_sentiment)
                # Jaga nada agar tidak menggurui
                softener = random.choice([
                    "Kalau cocok buatmu, ",
                    "Boleh dicoba pelan-pelan, ",
                    "Opsional ya, "
                ])
                response_text = response_text + (" " if response_text else "") + softener + reco

            # 8) Tambahkan rujukan krisis (paling akhir, singkat)
            if crisis_text:
                response_text = response_text + (" " if response_text else "") + crisis_text

            # 9) Mirror gaya user
            mirrored_response = self.mirror_user_style(response_text, style_analysis)

            # 10) Memory & counters
            self.update_short_term_memory(user_input, mirrored_response, current_sentiment)
            self.previous_sentiment = current_sentiment
            self.turn_count += 1

            # 11) JSON structure (TIDAK DIUBAH)
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
            print("🔄 Session reset")
        except Exception as e:
            print(f"Reset error: {e}")
