# chatbot/rasa_chatbot.py
import os, json, re, random, torch
import torch.nn.functional as F
from typing import Dict, Optional, List

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import google.generativeai as genai


class RasaChatbot:
    def __init__(self, model_name: str, gemini_api_key: str):
        print("v3 Initializing RasaChatbot (calibrated sentiment)…")
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
                print(f"⚠️ calibration.json not found on Hub ({fname}). Using defaults. Error: {e}")
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

        # 7) Response styles
        self.response_styles = {
            'sadness': {
                'level_1': [
                    "Kedengeran berat ya.",
                    "Hmm, kayaknya lagi nggak enak.", 
                    "Pasti susah ya.",
                    "Ada yang ganggu pikiran ya?",
                    "Lagi ada beban ya?",
                    "Hmm, tough day ya?",
                    "Kayaknya lagi down ya.",
                    "Seems like you're having a hard time.",
                    "Lagi nggak okay ya?"
                ],
                'level_2': [
                    "Kayaknya lagi susah banget ya. Mau cerita lebih lanjut?",
                    "Pasti berat banget rasanya sekarang.",
                    "Aku ada di sini kalau kamu butuh temen ngobrol.",
                    "It's okay to feel this way. Ada yang bisa aku bantu?",
                    "Kadang emang ada hari-hari yang berat. Gimana perasaanmu?",
                    "Aku ngerti ini pasti nggak mudah buat kamu.",
                    "Wanna talk about it? Sometimes sharing helps.",
                    "Take your time. Aku dengerin kok."
                ],
                'level_3': [
                    "Aku bisa ngebayangin betapa beratnya perasaan kamu. Kadang emang ada masa-masa kayak gini ya.",
                    "Thank you udah mau berbagi. Aku di sini buat dengerin kamu.",
                    "Perasaan kayak gini valid kok. Kamu nggak sendirian.",
                    "Aku appreciate kamu udah terbuka. Let's work through this together.",
                    "Sometimes life hits hard. Tapi percaya deh, this too shall pass.",
                    "I can feel how heavy this is for you. Dan it's okay untuk nggak okay."
                ]
            },
            'anger': {
                'level_1': [
                    "Sampe kesel gitu ya!",
                    "Hmm, bikin emosi banget.",
                    "Pantes sih kalau kesel.",
                    "That's frustrating!",
                    "Ugh, annoying banget ya.",
                    "Bikin gedeg ya?",
                    "I'd be mad too!"
                ],
                'level_2': [
                    "Sampe segitunya ya keselnya. Ada apa sih tadi?",
                    "Wajar banget kalau sampai marah gitu.",
                    "Damn, that must be really annoying. Cerita dong.",
                    "Aku ngerti kenapa kamu kesel. Wanna vent?",
                    "Sounds super frustrating. What triggered this?",
                    "That anger is justified. Mau ngeluarin unek-unek?"
                ],
                'level_3': [
                    "Aku ngerti banget kenapa kamu marah. Kadang emang ada hal yang bikin kita nggak bisa sabar.",
                    "Your anger makes total sense. Sometimes we just need to let it out.",
                    "Kemarahan kamu valid kok. Let's process this together.",
                    "I feel you. Some things just push our buttons, dan it's okay to be angry about it."
                ]
            },
            'fear': {
                'level_1': [
                    "Hmm, wajar sih kalau deg-degan.",
                    "Bikin khawatir ya.",
                    "Deg-degan gitu ya.",
                    "That's scary.",
                    "Pasti bikin anxious.",
                    "I get why you're worried.",
                    "Bikin was-was ya?"
                ],
                'level_2': [
                    "Pasti bikin nggak tenang banget ya. Emang ada apa?",
                    "Deg-degan kayak gini emang nggak enak.",
                    "Anxiety is no joke. What's making you worried?",
                    "Aku ngerti kekhawatiran kamu. Mau sharing?",
                    "That sounds nerve-wracking. Apa yang bikin takut?"
                ],
                'level_3': [
                    "Kekhawatiran kayak gitu emang bikin susah ya. Aku paham banget perasaan kamu.",
                    "Fear can be paralyzing. Tapi kamu nggak alone in this.",
                    "I understand your anxiety. Let's face this together, one step at a time.",
                    "Rasa takut itu manusiawi. Dan brave itu bukan berarti nggak takut, tapi tetap maju meski takut."
                ]
            },
            'happy': {
                'level_1': [
                    "Asik banget!",
                    "Seneng deh denger kamu happy.",
                    "Kedengeran seru ya.",
                    "Yay! Love the energy!",
                    "That's awesome!",
                    "Wah, good news ya?",
                    "Nice! Tell me more!"
                ],
                'level_2': [
                    "Seneng banget deh denger kamu udah lebih baik! Ada apa nih?",
                    "Asik banget sih, pasti lega ya.",
                    "Your happiness is contagious! Share the good vibes!",
                    "Love to see you happy! Spill the tea!",
                    "This energy! Cerita dong what made your day?"
                ],
                'level_3': [
                    "Aku seneng banget liat kamu bahagia kayak gini. Cerita dong apa yang bikin happy.",
                    "Your joy brings me joy too! Moments like these are precious.",
                    "This happiness looks good on you! Cherish these moments ya.",
                    "Seeing you this happy makes my day! Life's good when we can celebrate like this."
                ]
            },
            'love': {
                'level_1': [
                    "Kedengeran sayang banget ya.",
                    "Aww, sweet banget.",
                    "Gemes deh.",
                    "That's so cute!",
                    "Butterflies everywhere!",
                    "Love is in the air~",
                    "Ihiy, baper nih?"
                ],
                'level_2': [
                    "Kedengeran sayang banget sama dia. Gimana ceritanya?",
                    "Aww, pasti bikin hati anget ya.",
                    "The way you talk about them! So much love. Tell me everything!",
                    "Falling in love hits different ya? Cerita dong!",
                    "This is giving me all the feels! How did it start?"
                ],
                'level_3': [
                    "Perasaan sayang kayak gini emang indah ya. Mau cerita lebih tentang dia?",
                    "Love transforms everything, doesn't it? I can feel how special this person is to you.",
                    "Cinta emang bikin dunia lebih berwarna. Happy for you!",
                    "When you know, you know. Dan kayaknya you really know. Beautiful!"
                ]
            },
            'neutral': {
                'level_1': [
                    "Hmm, gitu ya.",
                    "Oh begitu.",
                    "I see.",
                    "Got it.",
                    "Interesting.",
                    "Hmm, okay.",
                    "Noted."
                ],
                'level_2': [
                    "Hmm, ada yang lagi kepikiran ya?",
                    "Gimana perasaan kamu tentang itu?",
                    "Interesting point. Elaborate dong?",
                    "I'm listening. Ada lagi?",
                    "Tell me more about that."
                ],
                'level_3': [
                    "Menarik juga. Aku penasaran sama pemikiran kamu tentang hal ini.",
                    "I sense there's more to this story. Wanna dive deeper?",
                    "Your perspective is intriguing. Help me understand better?",
                    "There's something on your mind. Feel free to explore it here."
                ]
            },
            # TAMBAHAN: Kategori greeting
            'greeting': {
                'level_1': [
                    "Hai! Gimana kabarnya?",
                    "Hello! Ada yang bisa aku bantu?",
                    "Halo! Seneng bisa ngobrol sama kamu.",
                    "Hi there! How's it going?",
                    "Halo! Apa kabar hari ini?",
                    "Hey! What's up?",
                    "Haii! Long time no see!"
                ]
            }
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

    # ============ Sentiment (calibrated) ============
    def analyze_sentiment(self, text: str) -> Dict:
        """Calibrated sentiment with confidence gate"""
        try:
            # TAMBAHAN: Deteksi greeting dulu
            text_lower = text.lower().strip()
            greeting_words = ['hai', 'halo', 'hello', 'hi', 'hey', 'pagi', 'siang', 'sore', 'malam', 'apa kabar', 'test', 'tes']
            
            # Jika pesan pendek dan mengandung greeting, return neutral
            if len(text.split()) <= 4 and any(word in text_lower for word in greeting_words):
                return {
                    'dominant_sentiment': 'neutral',
                    'confidence': 0.9,
                    'uncertain': False,
                    'is_greeting': True
                }
            
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
        
        # TAMBAHAN: Deteksi greeting di awal
        greeting_patterns = [
            r'^(hai|halo|hello|hi|hey|hei|pagi|siang|sore|malam|selamat)',
            r'^(apa kabar|gimana kabar|how are you|whats up|what\'s up)',
            r'^(test|tes|coba|ping|p)'
        ]
        
        for pattern in greeting_patterns:
            if re.search(pattern, user_lower):
                greetings = [
                    "Halo! Gimana kabarnya hari ini?",
                    "Hai! Ada yang bisa aku bantu?",
                    "Hello! Seneng bisa ngobrol sama kamu!",
                    "Halo! Gimana hari kamu?",
                    "Hi! Mau cerita apa nih?",
                    "Hey there! How's life treating you?",
                    "Haloo! What brings you here today?"
                ]
                return random.choice(greetings)
        
        # Existing special cases dengan variasi
        if user_lower in ['hmm', 'hm', 'mm']: 
            return random.choice([
                "Hmm, ada yang lagi kepikiran ya?",
                "Lagi mikirin apa nih?",
                "Ada yang mau diceritain?",
                "Something on your mind?"
            ])
        
        if user_lower in ['ok', 'oke', 'okay']: 
            return random.choice([
                "Oke. Ada yang mau diceritain lagi?",
                "Alright. What else?",
                "Got it. Anything else?",
                "Okay then. Lanjut?"
            ])
        
        if user_lower in ['ya', 'iya', 'yup', 'yep']: 
            return random.choice([
                "Hmm, gimana perasaan kamu sekarang?",
                "I see. Terus?",
                "Okay, and then?",
                "Alright, tell me more?"
            ])
        
        if re.match(r'^(wkwk|haha|hihi|hehe|lol|kwkw)+$', user_lower): 
            return random.choice([
                "Hahaha, ada yang lucu ya? Cerita dong!",
                "Ikutan ketawa ah! Ada apa nih?",
                "Seems fun! Share the joke!",
                "Love the energy! Spill!"
            ])
        
        if 'bingung' in user_lower:
            if style_analysis.get('pronouns') == 'saya':
                return "Kebingungan memang bikin nggak tenang. Mau coba diomongin pelan-pelan?"
            else:
                return random.choice([
                    "Hmm, bingung ya? Mau cerita apa yang bikin bingung?",
                    "Confusion is normal. Let's untangle it together?",
                    "Bingung gimana? Talk to me."
                ])
        
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
        
        # Kadang-kadang TIDAK mirror 100% untuk variasi (30% chance)
        if random.random() > 0.7:
            return response
        
        # Pronoun variation
        pronouns = style_analysis.get('pronouns', self.user_personality_profile['preferred_pronouns'])
        if pronouns == 'gue':
            # Variasikan antara gue, gw, w
            gue_variants = ['gue', 'gw', 'w']
            chosen = random.choice(gue_variants)
            response = response.replace('aku', chosen)
            response = response.replace('Aku', chosen.capitalize())
        elif pronouns == 'saya':
            response = response.replace('aku', 'saya')
            response = response.replace('Aku', 'Saya')
        
        # Exclamation variation - tidak selalu ganti semua
        ex_level = style_analysis.get('exclamation_level', 'low')
        if ex_level == 'high':
            if random.random() > 0.5:  # 50% chance
                sentences = response.split('. ')
                if len(sentences) > 1:
                    # Hanya ganti beberapa random, tidak semua
                    num_to_change = random.randint(1, min(2, len(sentences)))
                    indices = random.sample(range(len(sentences)), num_to_change)
                    for idx in indices:
                        if not sentences[idx].endswith('?'):
                            sentences[idx] = sentences[idx].rstrip('.!') + '!'
                    response = '. '.join(sentences)
        elif ex_level == 'medium' and response.count('!') == 0:
            if response.endswith('.') and random.random() > 0.5:
                response = response[:-1] + '!'
        
        # Repetition handling dengan variasi
        if style_analysis.get('uses_repetition'):
            repetition_words = {
                'banget': ['bangettt', 'bangeeet', 'bgt'],
                'asik': ['asikk', 'asyik', 'asiik'],
                'sedih': ['sedihh', 'sediih'],
                'kesel': ['keseel', 'keseell']
            }
            for word, variants in repetition_words.items():
                if word in response and random.random() > 0.5:
                    response = response.replace(word, random.choice(variants))
        
        # Tambahkan variasi slang
        if style_analysis.get('casualness', 0) > 2:
            slang_replacements = [
                ('tidak', random.choice(['nggak', 'gak', 'ga', 'kagak'])),
                ('sudah', random.choice(['udah', 'dah', 'uda'])),
                ('saja', random.choice(['aja', 'doang', 'ae'])),
                ('bagaimana', random.choice(['gimana', 'gmn', 'gimane'])),
                ('begitu', random.choice(['gitu', 'gt'])),
                ('dengan', random.choice(['sama', 'ama']))
            ]
            
            # Apply replacements randomly (not all)
            num_replacements = random.randint(1, min(3, len(slang_replacements)))
            selected_replacements = random.sample(slang_replacements, num_replacements)
            
            for formal, casual in selected_replacements:
                response = re.sub(r'\b' + formal + r'\b', casual, response, flags=re.IGNORECASE)
        
        # Emoji handling dengan variasi
        if style_analysis.get('uses_emoji') and self.user_personality_profile['emoji_usage']:
            if not re.search(r'[😀-🙏]', response) and random.random() > 0.3:  # 70% chance
                emoji_map = {
                    'happy': [' 😊', ' 😄', ' ☺️', ' 😁'],
                    'sadness': [' 😔', ' 😢', ' 💔', ' 😞'],
                    'love': [' 🥰', ' ❤️', ' 💕', ' 😍'],
                    'anger': [' 😤', ' 😠', ' 😑'],
                    'fear': [' 😰', ' 😟', ' 😨'],
                    'neutral': [' 🙂', ' 👍', ' ✨']
                }
                sentiment = getattr(self, 'current_sentiment', 'neutral')
                if sentiment in emoji_map:
                    response += random.choice(emoji_map[sentiment])
        
        return response

    def update_short_term_memory(self, user_input: str, bot_response: str, sentiment: str):
        self.short_term_memory.append({
            'user': user_input, 
            'bot': bot_response, 
            'sentiment': sentiment, 
            'turn': self.turn_count
        })
        # Perbesar memory untuk context yang lebih baik
        if len(self.short_term_memory) > 10:  # Dari 10 jadi 10 (atau bisa 15)
            self.short_term_memory.pop(0)

    def get_memory_context(self) -> str:
        if not self.short_term_memory: return ""
        parts = []
        for m in self.short_term_memory[-5:]:
            parts.append(f"User pernah bilang: '{m['user']}' (emosi: {m['sentiment']})")
        return "Konteks percakapan sebelumnya: " + " | ".join(parts) if parts else ""

    def detect_transition(self, current_sentiment: str) -> Optional[str]:
        if self.previous_sentiment and self.previous_sentiment != current_sentiment:
            return f"{self.previous_sentiment} → {current_sentiment}"
        return None
    
    def should_give_recommendation(self, user_input: str, current_sentiment: str) -> bool:
    # """Deteksi apakah user butuh rekomendasi"""
        user_lower = user_input.lower()
        
        # Explicit request patterns
        request_patterns = [
            r'\bsaran\b', r'\brekomendasi\b', r'\btips\b', r'\bsolusi\b',
            r'\bgimana ya\b', r'\bharus apa\b', r'\bapa yang harus\b',
            r'\btolong\b.*\bbantu\b', r'\bbantu\b.*\bdong\b',
            r'\bgimana cara\b', r'\bbiar\b.*\bbisa\b'
        ]
        
        # Check explicit request
        for pattern in request_patterns:
            if re.search(pattern, user_lower):
                return True
        
        # Check implicit need based on sentiment persistence
        if len(self.short_term_memory) >= 3:
            recent_sentiments = [m['sentiment'] for m in self.short_term_memory[-3:]]
            negative_sentiments = ['sadness', 'anger', 'fear']
            
            # If negative sentiment persists for 3+ turns
            if all(s in negative_sentiments for s in recent_sentiments):
                if self.turn_count >= 5:  # After building rapport
                    return True
        
        # Check for stuck patterns
        if self.turn_count >= 4:
            recent_messages = [m['user'].lower() for m in self.short_term_memory[-3:]]
            # If user keeps saying similar things (stuck in loop)
            if len(set(recent_messages)) == 1 or \
            all('bingung' in msg or 'gatau' in msg for msg in recent_messages):
                return True
                
        return False

    def generate_contextual_recommendation(self, user_input: str, current_sentiment: str) -> str:
    # """Generate recommendation using Gemini based on context"""
    
        # Build comprehensive context
        memory_context = ""
        if self.short_term_memory:
            conversations = []
            for m in self.short_term_memory[-5:]:
                conversations.append(f"User: {m['user']} (emosi: {m['sentiment']})")
            memory_context = "Percakapan sebelumnya:\n" + "\n".join(conversations)
        
        recommendation_prompt = f"""
    Berdasarkan konteks curhat berikut, berikan 1-2 rekomendasi yang sangat spesifik dan actionable.

    {memory_context}

    User terakhir bilang: "{user_input}"
    Emosi dominan: {current_sentiment}

    ATURAN REKOMENDASI:
    1. Sangat spesifik sesuai masalah user
    2. Praktis dan bisa langsung dilakukan
    3. Gunakan bahasa casual Indonesia
    4. Format: langsung rekomendasi tanpa pembuka formal
    5. Maksimal 2-3 kalimat
    6. Sesuaikan dengan emosi (jangan terlalu ceria kalau user sedih)
    7. Fokus pada small steps yang realistis

    Contoh format yang bagus:
    - Untuk sedih: "Coba deh mulai dengan nulis 3 hal kecil yang bikin kamu bersyukur hari ini di notes HP. Atau mungkin jalan kaki bentar 10 menit aja di sekitar rumah buat refresh pikiran."
    - Untuk marah: "Mungkin bisa coba teknik napas 4-7-8 dulu ya - tarik napas 4 detik, tahan 7 detik, buang 8 detik. Atau tulis semua yang bikin kesel di kertas terus sobek-sobek kertasnya."

    Berikan rekomendasi yang relevan dan membantu:
    """
        
        try:
            rec_response = self.chat_model.generate_content(recommendation_prompt)
            return rec_response.text.strip()
        except Exception as e:
            print(f"Error generating recommendation: {e}")
            # Fallback recommendations
            fallback = {
                'sadness': "Coba deh nulis perasaan kamu di jurnal atau notes HP. Kadang dengan nulis, beban jadi lebih ringan.",
                'anger': "Mungkin bisa coba ambil napas dalam-dalam dulu atau jalan kaki sebentar buat cooling down.",
                'fear': "Coba fokus ke hal-hal yang bisa kamu kontrol aja dulu. Bikin list kecil apa yang bisa kamu lakukan hari ini.",
                'happy': "Mantap! Mungkin bisa diabadikan momen ini di jurnal biar bisa jadi reminder kalau lagi down.",
                'love': "Sweet banget! Mungkin bisa ekspresikan perasaan ini dengan gesture kecil yang meaningful."
            }
            return fallback.get(current_sentiment, "Coba ambil waktu sejenak untuk refleksi diri ya.")

    def create_adaptive_prompt(self, user_input: str, current_sentiment: str, transition: Optional[str], style_analysis: Dict) -> str:
        empathy_level = self.get_empathy_level()
        
        # Deteksi jika greeting
        is_greeting = any(word in user_input.lower() for word in ['hai', 'halo', 'hello', 'hi', 'pagi', 'siang', 'sore', 'malam'])
        
        if is_greeting and self.turn_count == 0:
            current_sentiment = 'greeting'
        
        if current_sentiment in self.response_styles:
            style_options = self.response_styles[current_sentiment].get(f'level_{empathy_level}', 
                                                                    self.response_styles[current_sentiment].get('level_1', []))
        else:
            style_options = self.response_styles['neutral'][f'level_{empathy_level}']
        
        base_style = random.choice(style_options)
        
        memory_context = self.get_memory_context()
        transition_comment = self.create_natural_transition_comment(transition) if transition else ""
        
        # Variasi template
        template_variations = [
            "Berikan respons yang natural dan empatik.",
            "Respond dengan cara yang supportive dan understanding.",
            "Jawab dengan hangat dan penuh perhatian.",
            "Berikan respons yang genuine dan relatable.",
            "Respond secara authentic dan caring."
        ]
        
        base_instruction = random.choice(template_variations)
        
        # Personality traits random
        personality_traits = random.choice([
            "friendly dan supportive",
            "warm dan understanding", 
            "caring dan attentive",
            "empathetic dan genuine",
            "thoughtful dan kind"
        ])

        style_instructions = f"""
    Gaya komunikasi user:
    - Pronouns: {style_analysis.get('pronouns', 'aku')}
    - Exclamation level: {style_analysis.get('exclamation_level', 'low')}
    - Uses repetition: {style_analysis.get('uses_repetition', False)}
    - Uses emoji: {style_analysis.get('uses_emoji', False)}
    - Slang words: {', '.join(style_analysis.get('slang_words', []))}
    - Message length: {style_analysis.get('message_length', 'medium')}
    """

        should_recommend = self.should_give_recommendation(user_input, current_sentiment)
        recommendation_note = ""
        if should_recommend:
            recommendation_note = """
    Note: User sepertinya butuh saran/rekomendasi. Tapi JANGAN berikan saran di response ini, 
    karena akan digenerate terpisah. Fokus pada empati dan validasi perasaan saja.
    """

        prompt = f"""
    Kamu adalah chatbot yang {personality_traits}.

    {memory_context}

    User berkata: "{user_input}"
    Emosi user: {current_sentiment}
    Turn ke: {self.turn_count + 1}

    {style_instructions}

    Base response style sebagai inspirasi (JANGAN copy paste): "{base_style}"

    {f"Transition comment: {transition_comment}" if transition_comment else ""}

    {recommendation_note}

    {base_instruction}

    PENTING - VARIASI RESPONS:
    - JANGAN gunakan pola kalimat yang sama berulang-ulang
    - JANGAN selalu mulai dengan "Hmm" atau "Aku ngerti"
    - Variasikan struktur kalimat:
    * Kadang mulai dengan pertanyaan
    * Kadang langsung statement
    * Kadang exclamation
    * Kadang dengan observation
    - Mix bahasa Indonesia dan sedikit English untuk natural feel
    - Gunakan berbagai cara untuk menunjukkan empati, BUKAN hanya "aku ngerti" atau "pasti berat"
    - Sesekali gunakan humor ringan jika appropriate
    - Be creative dan spontan, HINDARI template responses!

    Contoh variasi opening yang bagus:
    - "Wah, that sounds..."
    - "Okay, jadi..."
    - Langsung pertanyaan: "Sejak kapan..."
    - "Actually, itu wajar kok..."
    - Atau langsung ke point tanpa basa-basi

    INSTRUKSI AKHIR:
    1. Respons maksimal 2-3 kalimat
    2. Natural dan conversational
    3. Mirror style user tapi dengan variasi
    4. JANGAN monoton atau repetitif
    """
        
        return prompt

    def chat(self, user_input: str) -> Dict:
        try:
            # TAMBAHAN: Reset/shuffle responses setiap beberapa turn untuk freshness
            if self.turn_count % 5 == 0 and self.turn_count > 0:
                for sentiment in self.response_styles:
                    for level in self.response_styles[sentiment]:
                        if isinstance(self.response_styles[sentiment][level], list):
                            random.shuffle(self.response_styles[sentiment][level])
        
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
            
            # logika rekomendasi
            should_recommend = self.should_give_recommendation(user_input, current_sentiment)
            if should_recommend:
                recommendation = self.generate_contextual_recommendation(user_input, current_sentiment)
                # Integrate recommendation naturally into response
                response_text = f"{response_text} {recommendation}"
            mirrored_response = self.mirror_user_style(response_text, style_analysis)

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
            print("Session reset")
        except Exception as e:
            print(f"Reset error: {e}")

