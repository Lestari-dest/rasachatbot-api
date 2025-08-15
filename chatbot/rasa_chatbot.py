# chatbot/rasa_chatbot.py
import os, json, re, random, torch
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
from collections import Counter

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import google.generativeai as genai


class RasaChatbot:
    def __init__(self, model_name: str, gemini_api_key: str):
        print("v2.2 Initializing RasaChatbot (humanized, contextual memory)…")
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

        # Perbesar memori percakapan (bisa override via env MEMORY_LIMIT)
        self.memory_limit = int(os.getenv("MEMORY_LIMIT", "10"))
        self.short_term_memory: List[Dict] = []  # simpan hingga memory_limit
        self.thread_summary: str = ""            # ringkasan percakapan (diupdate berkala)
        self.topic_counter: Counter = Counter()  # hitung topik

        self.user_personality_profile = {
            'preferred_pronouns': 'aku',
            'formality_level': 'casual',
            'emoji_usage': False,
            'exclamation_tendency': 'low',
            'openness_level': 1
        }
        self.confidence_threshold = 0.65

        # 7) Response styles (tone bank)
        self.response_styles = {
            'sadness': {
                'level_1': ["Kedengeran berat ya.", "Hmm, rasanya lagi nggak enak.", "Pasti nggak mudah ya."],
                'level_2': ["Sepertinya lagi berat banget. Mau cerita pelan aja?", "Pasti rasanya ketarik ke bawah ya."],
                'level_3': ["Aku bisa ngebayangin capeknya di posisi kamu. Kita bisa bahas pelan sesuai ritme kamu."]
            },
            'anger': {
                'level_1': ["Sampai bikin kesal, ya.", "Wajar kok ngerasa jengkel.", "Aku denger kok kalau ini bikin emosi."],
                'level_2': ["Keselnya kebawa banget ya. Emang kejadian apa?", "Aku paham kenapa ini bikin panas kepala."],
                'level_3': ["Rasanya pengen meledak; kita jaga jarak aman dulu dari sumbernya, lalu lihat pelan apa yang bisa dikendalikan."]
            },
            'fear': {
                'level_1': ["Deg-degan gitu ya.", "Bikin khawatir, ya.", "Rasanya nggak tenang."],
                'level_2': ["Khawatirnya kerasa di badan, ya. Boleh cerita apa pemicunya?", "Deg-degannya cukup kuat ya."],
                'level_3': ["Kekhawatiran bisa terasa nyata; kita coba lihat bagian kecil yang bisa dipegang dulu."]
            },
            'happy': {
                'level_1': ["Asik!", "Ikut seneng dengernya.", "Kedengeran lega."],
                'level_2': ["Seneng banget denger kabar baikmu! Ada apa nih?", "Lega ya, enak banget rasanya."],
                'level_3': ["Momen begini enak disimpan. Cerita sedikit biar aku ikut merayakan."]
            },
            'love': {
                'level_1': ["Kedengeran hangat, ya.", "Manis juga ceritanya.", "Gemes bacanya."],
                'level_2': ["Kayaknya kamu peduli banget. Gimana dinamika kalian belakangan?", "Ada yang bikin hati hangat, ya."],
                'level_3': ["Rasa sayang itu berharga; pelan-pelan cari ritme yang nyaman buat kalian berdua."]
            },
            'neutral': {
                'level_1': ["Oke, aku dengerin.", "Iya, aku nangkep.", "Baik, noted."],
                'level_2': ["Ada yang lagi nyangkut di pikiran, ya?", "Gimana rasanya pas kejadian itu?"],
                'level_3': ["Menarik. Kita bisa bongkar bagian kecilnya dulu kalau kamu mau."]
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

        # 9) Pemetaan topik → rekomendasi kontekstual
        self.topic_patterns = {
            'tidur': [r'\bsusah\s*tidur\b', r'\binsomnia\b', r'\bnggak\s*bisa\s*tidur\b', r'\btidur\b'],
            'makan': [r'\bnafsu\s*makan\b', r'\bmakan\b'],
            'kerja': [r'\bkerja(an)?\b', r'\boffice\b', r'\bbos\b'],
            'kuliah': [r'\bkuliah\b', r'\bkampus\b', r'\bskripsi\b', r'\btugas\b', r'\bdeadline\b'],
            'hubungan': [r'\bpacar\b', r'\bpasangan\b', r'\bteman\b', r'\bkeluarga\b', r'\brumah\b', r'\bkonflik\b', r'\bertengkar\b'],
            'kehilangan': [r'\bkehilangan\b', r'\bmeninggal\b', r'\bduka\b'],
            'cemas_panik': [r'\bpanik\b', r'\bserangan\s*panik\b', r'\bnapas\b', r'\bdeg(-| )?degan\b', r'\bwas(-| )?was\b', r'\bcemas\b'],
            'overthinking': [r'\boverthinking\b', r'\bmuter\b', r'\bkepikiran\b'],
            'keuangan': [r'\bduit\b', r'\buang\b', r'\bkeuangan\b', r'\bbayar\b', r'\btagihan\b'],
            'media': [r'\bscroll\b', r'\bdoomscroll\b', r'\bberita\b', r'\bmedsos\b', r'\btiktok\b', r'\binstagram\b'],
            'kesehatan_umum': [r'\bsakit\b', r'\bsehat\b', r'\bpusing\b', r'\bmual\b', r'\bcapek\b', r'\blelah\b'],
        }

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
        t = re.sub(r'[^\w\s!?.,:;()\-]+', ' ', t)
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
                "Seneng denger kamu lebih lega.",
                "Asik, pelan-pelan ngerasa lebih baik ya."
            ]),
            'anger → happy': random.choice([
                "Syukur bisa lebih adem sekarang.",
                "Asik, rasanya udah nggak seketat tadi ya."
            ]),
            'anger → sadness': random.choice([
                "Dari kesal jadi terasa sedih, ya.",
                "Sekarang beratnya lebih ke sedih."
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
                "Hangat sempat ada, terus jadi berat lagi ya.",
            ])
        }
        return transition_map.get(transition, "")

    def handle_special_cases(self, user_input: str, style_analysis: Dict) -> Optional[str]:
        user_lower = user_input.lower().strip()
        if user_lower in ['hmm', 'hm', 'mm']: return "Hmm, ada yang lagi kepikiran ya?"
        if user_lower in ['ok', 'oke', 'okay']: return "Oke. Ada yang mau diceritain lagi?"
        if user_lower in ['ya', 'iya', 'yup', 'yep']: return "Hmm, gimana perasaanmu sekarang?"
        if re.match(r'^(wkwk|haha|hihi|hehe|lol|kwkw)+$', user_lower): return "Hahaha, ada yang lucu ya? Cerita dong!"
        if 'bingung' in user_lower:
            return "Rasa bingung bikin nggak tenang. Mau kita urai pelan dari bagian paling kecil?"
        love_confusion_patterns = [r'sayang.*tapi.*gak tau', r'cinta.*bingung', r'suka.*gimana']
        for p in love_confusion_patterns:
            if re.search(p, user_lower):
                return "Kedengeran kamu peduli, tapi ragu langkahnya. Kita obrolin bareng, ya?"
        return None

    def get_empathy_level(self) -> int:
        openness = self.user_personality_profile['openness_level']
        if self.turn_count <= 2: return 1
        elif self.turn_count <= 5 or openness <= 2: return 2
        else: return 3

    # ======== Anti repetisi & frasa jarak sehat ========
    def _gentle_distance_phrase(self) -> str:
        # Hindari “berkenan” / “buru-buru”
        choices = [
            "Kalau kamu mau, kita mulai pelan aja.",
            "Boleh cerita sepotong dulu, senyaman kamu.",
            "Aku dengerin, ritmenya ikut kamu."
        ]
        return random.choice(choices if self.turn_count <= 2 else choices + [
            "Ambil tempo yang paling nyaman buatmu."
        ])

    def _dedupe_tokens(self, text: str) -> str:
        """Hapus kata berurutan yang sama & bigram berulang sederhana."""
        # hapus kata kembar berurutan
        words = text.split()
        result = []
        prev = None
        for w in words:
            if prev and w.lower() == prev.lower():
                continue
            result.append(w)
            prev = w
        text = " ".join(result)
        # hapus bigram berulang berurutan
        tokens = text.split()
        cleaned = []
        i = 0
        while i < len(tokens):
            if i+3 < len(tokens) and tokens[i:i+2] == tokens[i+2:i+4]:
                cleaned.extend(tokens[i:i+2])
                i += 4
            else:
                cleaned.append(tokens[i])
                i += 1
        return " ".join(cleaned)

    # ======== Micro story (jarang) ========
    def _maybe_micro_story(self, sentiment: str) -> Optional[str]:
        chance = 0.10 if self.turn_count <= 2 else 0.20
        if random.random() > chance:
            return None
        bank = {
            'sadness': [
                "Dulu aku sempat ngerasa hari-hari gelap; aku mulai dari hal kecil—merapikan satu sudut meja—aneh tapi bikin ada rasa pegang kendali.",
                "Pernah juga rasanya berat; jalan sore tanpa headset bantu aku ngedengerin isi kepala sendiri."
            ],
            'anger': [
                "Aku pernah kebawa emosi; napas 4-4-6 nolong aku buat nggak meledak.",
                "Waktu kesel sama orang dekat, aku nulis draft dulu biar nggak nyesel habis kirim."
            ],
            'fear': [
                "Aku kenal rasa was-was; menamai 5 benda yang kulihat bikin tegangnya turun.",
                "Pernah juga deg-degan tanpa alasan jelas; ngecek 'apa fakta yang beneran terjadi' bikin tanahnya kerasa lagi."
            ],
            'happy': [
                "Momen kecil yang enak gini suka jadi bahan bakar buat besok.",
                "Nyimpen satu kalimat syukur bikin aku punya jangkar pas hari lagi turun."
            ],
            'love': [
                "Rasa hangat pernah bikin aku berani jujur pelan; ternyata ritme yang nyaman itu kunci.",
                "Aku pernah deg-degan manis; nulis tiga apresiasi bikin rasanya makin jelas."
            ],
            'neutral': [
                "Hari yang datar kadang enak buat nyusun langkah; aku suka mulai tugas 5 menit biar mesin nyala.",
                "Saat nggak ada apa-apa, aku pilih satu hal kecil yang bisa selesai sekarang."
            ]
        }
        return random.choice(bank.get(sentiment, bank['neutral']))

    # ======== Topik & rekomendasi kontekstual ========
    def _extract_topics(self, text: str) -> List[str]:
        t = text.lower()
        hits = []
        for topic, patterns in self.topic_patterns.items():
            if any(re.search(p, t) for p in patterns):
                hits.append(topic)
        return hits

    def _contextual_recommendation(self, topics: List[str], sentiment: str) -> Optional[str]:
        """Pilih saran yang relevan dengan topik; fallback ke emosi jika topik kosong."""
        reco_bank = {
            'tidur': [
                "Coba matikan layar 30 menit sebelum tidur dan bikin lampu remang.",
                "Atur jam tidur bangun yang sama selama 3 hari berturut-turut."
            ],
            'makan': [
                "Kalau selera turun, coba porsi kecil tapi sering, plus air hangat.",
                "Siapkan camilan sederhana berprotein biar energi nggak anjlok."
            ],
            'kerja': [
                "Timebox 25 menit fokus + 5 menit jeda untuk satu tugas paling kecil.",
                "Mulai dari tugas berdampak tapi ringan; tandai selesai biar ada momentum."
            ],
            'kuliah': [
                "Bikin outline 3 poin, lalu kerjakan 1 paragraf saja dulu.",
                "Pisahkan ‘riset’ dan ‘nulis’; set timer 20 menit per mode."
            ],
            'hubungan': [
                "Kalau ada konflik, coba pakai kalimat ‘Aku merasa… saat…; kebutuhanku…’.",
                "Coba janji waktu ngobrol 15 menit, satu bicara—satu mendengar."
            ],
            'kehilangan': [
                "Kalau pas, tulis surat pendek untuk orang/hal yang kamu rindukan.",
                "Buat momen kecil mengenang (mis. nyalakan lilin/putar lagu favorit)."
            ],
            'cemas_panik': [
                "Coba grounding 5-4-3-2-1 (lihat, raba, dengar, cium, rasa) perlahan.",
                "Napas 4-4-6 lima kali bisa bantu redakan sinyal tubuh."
            ],
            'overthinking': [
                "Jadwalkan ‘worry time’ 10 menit, di luar itu catat dan tunda.",
                "Tantang pikiran dengan ‘Apa buktinya? Ada alternatif lain?’"
            ],
            'keuangan': [
                "Tuliskan 3 pengeluaran utama hari ini dan satu langkah kecil mengurangi satu di antaranya.",
                "Pakai aturan 24 jam sebelum beli non-darurat."
            ],
            'media': [
                "Batasi scroll ke 15 menit, lalu cek ulang rasanya di tubuh.",
                "Taruh ponsel di luar jangkauan selama 20 menit saat fokus."
            ],
            'kesehatan_umum': [
                "Minum air, gerakkan badan 2 menit, dan tarik napas dalam 3 kali.",
                "Kalau badan memberi sinyal lelah, izinkan istirahat singkat terjadwal."
            ],
        }
        if topics:
            topic = topics[0]
            return random.choice(reco_bank.get(topic, [])) if reco_bank.get(topic) else None

        # fallback by sentiment kalau tidak ada topik jelas
        fallback = {
            'sadness': "Kalau cocok, coba grounding 5-4-3-2-1 sebentar.",
            'anger': "Tunda balasan dengan draft, kirim setelah 30 menit.",
            'fear': "Ucapkan 3 hal yang benar-benar kamu lihat/dengar/rasakan sekarang.",
            'happy': "Catat satu hal yang bikin kamu bangga hari ini.",
            'love': "Kalau mau, tulis pesan jujur yang ringan dan spesifik.",
            'neutral': "Pilih tugas 5 menit untuk mulai gerakin mesin."
        }
        return fallback.get(sentiment)

    def _should_attach_reco(self, sentiment: str, has_topic: bool) -> bool:
        base = {'sadness': 0.6, 'anger': 0.6, 'fear': 0.6, 'neutral': 0.3, 'love': 0.25, 'happy': 0.25}
        rate = base.get(sentiment, 0.3)
        if self.turn_count <= 1:
            rate *= 0.5
        if has_topic:
            rate += 0.15
        return random.random() < min(max(rate, 0.1), 0.8)

    # ======== Krisis detection & referral (lembut) ========
    def _detect_crisis(self, text: str) -> bool:
        t = text.lower()
        return any(re.search(p, t) for p in self.CRISIS_PATTERNS)

    def _crisis_referral(self, style_analysis: Dict) -> str:
        pron = style_analysis.get('pronouns', 'aku')
        lines = [
            "Aku khawatir sama keselamatanmu.",
            "Kalau kamu merasa berisiko melukai diri atau ada bahaya sekarang, minta bantuan langsung ya.",
            "Hubungi layanan darurat setempat, tenaga kesehatan jiwa, atau orang tepercaya di dekatmu."
        ]
        if pron == 'saya':
            lines[0] = "Saya khawatir dengan keselamatan Anda."
        return " ".join(lines)

    # ======== Prompt builder ========
    def create_adaptive_prompt(self, user_input: str, current_sentiment: str, transition: Optional[str], style_analysis: Dict, topics: List[str]) -> str:
        empathy_level = self.get_empathy_level()
        if current_sentiment in self.response_styles:
            style_options = self.response_styles[current_sentiment][f'level_{empathy_level}']
        else:
            style_options = self.response_styles['neutral'][f'level_{empathy_level}']
        base_style = random.choice(style_options)

        memory_context = self.get_memory_context()
        thread_summary = f"Ringkasan: {self.thread_summary}" if self.thread_summary else ""
        transition_comment = self.create_natural_transition_comment(transition) if transition else ""

        recipes = [
            "Validasi hangat 1 kalimat, lanjut 1 pertanyaan terbuka ringan.",
            "Parafrase inti cerita user dengan kata berbeda + sebutkan perasaan yang mungkin dirasakan.",
            "Tunjukkan perhatian ke sinyal tubuh/energi, beri ruang tanpa menekan.",
            "Jawab pendek tapi hangat; jangan memaksa user menjelaskan panjang.",
            "Mulai dari momen kecil yang aman/neutral sebelum sentuh inti masalah."
        ]
        picked_recipe = random.choice(recipes)

        style_instructions = f"""
GAYA USER:
- Pronoun: {style_analysis.get('pronouns', 'aku')}
- Tanda seru: {style_analysis.get('exclamation_level', 'low')}
- Repetisi: {style_analysis.get('uses_repetition', False)}
- Emoji: {style_analysis.get('uses_emoji', False)}
- Slang: {', '.join(style_analysis.get('slang_words', [])) or '-'}
- Panjang pesan user: {style_analysis.get('message_length', 'medium')}

TOPIK TERDETEKSI: {', '.join(topics) if topics else '(belum jelas)'}
ATURAN PANJANG:
- Pesan pendek → 1 kalimat (≤16 kata).
- Pesan sedang → 1–2 kalimat.
- Pesan panjang → 2–4 kalimat.
HINDARI:
- Frasa “buru-buru”, “berkenan”, dan pengulangan frasa yang sama dari respons sebelumnya.
- Bullet/angka di output.
RESEP RESPON (acak): {picked_recipe}
REKOMENDASI:
- Jika perlu, beri 1 saran aman, relevan dengan topik, tidak menggurui.
"""

        prompt = f"""
Kamu adalah pendengar empatik yang natural (tidak kaku, terasa manusiawi).

{memory_context}
{thread_summary}

User: "{user_input}"
Emosi dominan: {current_sentiment}
Empathy level: {empathy_level}/3
Turn ke: {self.turn_count + 1}

Base style cue: "{base_style}"
{f"Catat perubahan emosi: {transition_comment}" if transition_comment else ""}

{style_instructions}

OUTPUT:
- Tulis respons final saja (tanpa label).
- Selipkan kalimat ini jika cocok sebagai pembuka jarak sehat: "{self._gentle_distance_phrase()}"
"""
        return prompt

    # ======== Memory helpers ========
    def update_short_term_memory(self, user_input: str, bot_response: str, sentiment: str):
        topics = self._extract_topics(user_input)
        self.topic_counter.update(topics)
        self.short_term_memory.append({
            'user': user_input, 'bot': bot_response,
            'sentiment': sentiment, 'turn': self.turn_count, 'topics': topics
        })
        # batasi sesuai memory_limit
        while len(self.short_term_memory) > self.memory_limit:
            old = self.short_term_memory.pop(0)
            # kurangi counter topik yang keluar dari jendela memori
            for t in old.get('topics', []):
                if self.topic_counter[t] > 0:
                    self.topic_counter[t] -= 1

        # update ringkasan setiap 3 turn agar konteks rapih
        if self.turn_count % 3 == 2:
            try:
                snippet = "\n".join([f"U:{m['user']}\nB:{m['bot']}" for m in self.short_term_memory[-6:]])
                prompt = f"Ringkas percakapan berikut menjadi 1-2 kalimat yang menangkap tema/tujuan tanpa detail sensitif:\n{snippet}"
                s = self.chat_model.generate_content(prompt)
                self.thread_summary = (s.text or "").strip()[:300]
            except Exception:
                pass  # kalau gagal, biarkan summary lama

    def get_memory_context(self) -> str:
        if not self.short_term_memory: return ""
        last = self.short_term_memory[-3:]  # ambil 3 interaksi terakhir biar cukup konteks
        parts = []
        for m in last:
            # pendek dan fokus
            u = m['user']
            u = (u[:120] + '…') if len(u) > 120 else u
            parts.append(f"User dulu bilang: '{u}' (emosi: {m['sentiment']})")
        return "Konteks singkat: " + " | ".join(parts)

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

        # Repetisi user
        if style_analysis.get('uses_repetition'):
            response = re.sub(r'\bbanget\b', 'bangettt', response)
            response = re.sub(r'\basik\b', 'asikk', response)

        # Emoji (opsional)
        if style_analysis.get('uses_emoji') and self.user_personality_profile['emoji_usage']:
            if not re.search(r'[😀-🙏]', response):
                emoji_map = {'happy':' 😊','sadness':' 😔','love':' 🥰','anger':' 😤','fear':' 😰','neutral':' 🙂'}
                response += emoji_map.get(getattr(self, 'current_sentiment', 'neutral'), '')

        # Panjang respons menyesuaikan panjang user
        msg_len = style_analysis.get('message_length', 'medium')
        if msg_len == 'short':
            response = re.split(r'(?<=[.!?])\s+', response)[0]

        # Anti repetisi akhir
        response = self._dedupe_tokens(response)
        # Hapus frasa yang dihindari
        for bad in ["buru-buru", "buru-buru", "berkenan"]:
            response = re.sub(rf'\b{bad}\b', '', response, flags=re.IGNORECASE)
            response = re.sub(r'\s{2,}', ' ', response).strip()

        return response

    # ======== Chat utama (JSON tetap sama) ========
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

            # 1) Sentiment dari model (WAJIB)
            sentiment_result = self.analyze_sentiment(user_input)
            current_sentiment = sentiment_result['dominant_sentiment']
            self.current_sentiment = current_sentiment

            # 2) Update gaya & transisi
            self.update_personality_profile(style_analysis)
            transition = self.detect_transition(current_sentiment)

            # 3) Krisis check
            crisis_text = None
            if self._detect_crisis(user_input):
                crisis_text = self._crisis_referral(style_analysis)

            # 4) Topik sekarang + dari memori (prioritaskan yang terbaru)
            current_topics = self._extract_topics(user_input)
            # Tambahkan topik kuat dari memori bila kosong
            if not current_topics and self.topic_counter:
                top_mem = [t for t, _ in self.topic_counter.most_common(1)]
                current_topics = top_mem

            # 5) Prompt adaptif (kaya konteks)
            adaptive_prompt = self.create_adaptive_prompt(user_input, current_sentiment, transition, style_analysis, current_topics)
            resp = self.chat_session.send_message(adaptive_prompt)
            response_text = (resp.text or "").strip()

            # 6) Micro story (opsional)
            micro = self._maybe_micro_story(current_sentiment)
            if micro:
                if style_analysis.get('message_length') == 'short':
                    response_text = re.split(r'(?<=[.!?])\s+', response_text)[0]
                if response_text and not re.search(r'[.!?]\s*$', response_text):
                    response_text += '.'
                response_text = response_text + " " + micro

            # 7) Rekomendasi kontekstual (nyambung topik)
            if self._should_attach_reco(current_sentiment, bool(current_topics)):
                reco = self._contextual_recommendation(current_topics, current_sentiment)
                if reco:
                    softener = random.choice(["Kalau cocok buatmu, ", "Boleh dicoba santai, ", "Opsional ya, "])
                    if response_text and not re.search(r'[.!?]\s*$', response_text):
                        response_text += '.'
                    response_text += " " + softener + reco

            # 8) Rujukan krisis di akhir (jika ada)
            if crisis_text:
                if response_text and not re.search(r'[.!?]\s*$', response_text):
                    response_text += '.'
                response_text += " " + crisis_text

            # 9) Mirror gaya & anti repetisi
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
            self.thread_summary = ""
            self.topic_counter = Counter()
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
