import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import google.generativeai as genai
import random
import re
from datetime import datetime
from typing import Dict, Optional, List

class RasaChatbot:
    def __init__(self, model_name: str, gemini_api_key: str):
        print("🚀 Initializing Enhanced RasaChatbot...")

        # Load IndoBERT sentiment model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # self.labels = ['sadness', 'love', 'anger', 'happy', 'fear']
        # NEW (CORRECT):
        self.labels = ['love', 'anger', 'sadness', 'happy', 'fear']

        # Gemini
        genai.configure(api_key=gemini_api_key)
        self.chat_model = genai.GenerativeModel("gemini-2.0-flash")
        self.chat_session = self.chat_model.start_chat(history=[])

        # Enhanced Memory System
        self.previous_sentiment = None
        self.turn_count = 0
        self.short_term_memory = []  # Last 3 messages
        self.user_personality_profile = {
            'preferred_pronouns': 'aku',  # default
            'formality_level': 'casual',
            'emoji_usage': False,
            'exclamation_tendency': 'low',
            'openness_level': 1  # 1-5, starts low
        }

        # Confidence threshold for sentiment
        self.confidence_threshold = 0.65

        # Enhanced response styles with progression levels - More human-like
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

        print("✅ Enhanced RasaChatbot siap dengan fitur adaptasi lengkap!")

    def analyze_sentiment(self, text: str) -> Dict:
        """Enhanced sentiment analysis with confidence check"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidence, predicted = torch.max(scores, dim=1)
            
            # Confidence check + fallback neutral
            if confidence.item() < self.confidence_threshold:
                return {
                    'dominant_sentiment': 'neutral',  # fallback to neutral
                    'confidence': confidence.item(),
                    'uncertain': True
                }
            
            return {
                'dominant_sentiment': self.labels[predicted.item()],
                'confidence': confidence.item(),
                'uncertain': False
            }
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return {'dominant_sentiment': 'neutral', 'confidence': 0.5, 'uncertain': True}

    def analyze_user_style(self, user_input: str) -> Dict:
        """Enhanced style analysis with detailed slang adaptation"""
        style_profile = {}
        
        # Pronoun detection (gue/aku/saya)
        if re.search(r'\b(gue|gw)\b', user_input.lower()):
            style_profile['pronouns'] = 'gue'
        elif re.search(r'\b(aku|ak)\b', user_input.lower()):
            style_profile['pronouns'] = 'aku'
        elif re.search(r'\bsaya\b', user_input.lower()):
            style_profile['pronouns'] = 'saya'
        else:
            style_profile['pronouns'] = self.user_personality_profile['preferred_pronouns']
        
        # Exclamation and repetition patterns
        exclamation_count = user_input.count('!')
        if exclamation_count >= 3:
            style_profile['exclamation_level'] = 'high'
        elif exclamation_count >= 1:
            style_profile['exclamation_level'] = 'medium'
        else:
            style_profile['exclamation_level'] = 'low'
        
        # Repeated letters (bangettt, sedihhhh)
        repeated_pattern = re.findall(r'(\w)\1{2,}', user_input.lower())
        style_profile['uses_repetition'] = len(repeated_pattern) > 0
        
        # Emoji usage
        emoji_pattern = re.findall(r'[😀-🙏]', user_input)
        style_profile['uses_emoji'] = len(emoji_pattern) > 0
        
        # Slang words detection
        slang_words = ['wkwk', 'anjir', 'sih', 'banget', 'dong', 'deh', 'kok', 'kayak', 'gimana']
        found_slang = [word for word in slang_words if word in user_input.lower()]
        style_profile['slang_words'] = found_slang
        style_profile['casualness'] = len(found_slang)
        
        # Message length
        word_count = len(user_input.split())
        if word_count <= 3:
            style_profile['message_length'] = 'short'
        elif word_count <= 10:
            style_profile['message_length'] = 'medium'
        else:
            style_profile['message_length'] = 'long'
        
        return style_profile

    def update_personality_profile(self, style_analysis: Dict):
        """Update user personality profile based on consistent patterns"""
        # Update preferred pronouns
        if style_analysis.get('pronouns'):
            self.user_personality_profile['preferred_pronouns'] = style_analysis['pronouns']
        
        # Update emoji usage
        if style_analysis.get('uses_emoji'):
            self.user_personality_profile['emoji_usage'] = True
        
        # Update exclamation tendency
        self.user_personality_profile['exclamation_tendency'] = style_analysis.get('exclamation_level', 'low')
        
        # Update openness level based on message length and turn count
        if self.turn_count > 3 and style_analysis.get('message_length') == 'long':
            self.user_personality_profile['openness_level'] = min(5, self.user_personality_profile['openness_level'] + 1)

    def create_natural_transition_comment(self, transition: str) -> str:
        """Create natural emotional transition acknowledgment - More human-like"""
        transition_map = {
            'sadness → happy': random.choice([
                "Seneng deh denger kamu udah lebih baik!",
                "Asik banget! Kedengeran udah mulai membaik ya.",
                "Alhamdulillah ya udah mulai happy lagi."
            ]),
            'anger → happy': random.choice([
                "Asik banget! Seneng deh denger kamu udah lebih lega.",
                "Kedengeran udah nggak kesel lagi ya. Seneng banget!",
                "Alhamdulillah ya udah mulai happy lagi."
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
        """Handle special input cases with natural responses"""
        user_lower = user_input.lower().strip()
        
        # Very short responses
        if user_lower in ['hmm', 'hm', 'mm']:
            return "Hmm, ada yang lagi kepikiran ya?"
        
        if user_lower in ['ok', 'oke', 'okay']:
            return "Oke. Ada yang mau diceritain lagi?"
        
        if user_lower in ['ya', 'iya', 'yup', 'yep']:
            return "Hmm, gimana perasaan kamu sekarang?"
        
        # Laughter patterns - Fixed regex
        if re.match(r'^(wkwk|haha|hihi|hehe|lol|kwkw)+$', user_lower):
            return "Hahaha, ada yang lucu ya? Cerita dong!"
        
        # Confusion expressions
        if 'bingung' in user_lower:
            if style_analysis.get('pronouns') == 'saya':
                return "Kebingungan memang bikin nggak tenang. Mau coba diomongin pelan-pelan?"
            else:
                return "Hmm, bingung ya? Mau cerita apa yang bikin bingung?"
        
        # Love/relationship mentions with confusion
        love_confusion_patterns = [
            r'sayang.*tapi.*gak tau',
            r'cinta.*bingung',
            r'suka.*gimana'
        ]
        for pattern in love_confusion_patterns:
            if re.search(pattern, user_lower):
                return "Kedengeran sayang banget ya sama dia, tapi bingung juga harus gimana. Mau ngobrolin bareng-bareng?"
        
        return None

    def get_empathy_level(self) -> int:
        """Determine empathy progression level (1-3)"""
        openness = self.user_personality_profile['openness_level']
        if self.turn_count <= 2:
            return 1
        elif self.turn_count <= 5 or openness <= 2:
            return 2
        else:
            return 3

    def mirror_user_style(self, base_response: str, style_analysis: Dict) -> str:
        """Mirror user's communication style"""
        response = base_response
        
        # Mirror pronoun usage
        pronouns = style_analysis.get('pronouns', self.user_personality_profile['preferred_pronouns'])
        if pronouns == 'gue':
            response = response.replace('aku', 'gue')
        elif pronouns == 'saya':
            response = response.replace('aku', 'saya')
        
        # Mirror exclamation level
        exclamation_level = style_analysis.get('exclamation_level', 'low')
        if exclamation_level == 'high' and '.' in response:
            response = response.replace('.', '!')
        elif exclamation_level == 'medium' and response.count('!') == 0:
            # Add one exclamation at the end
            if response.endswith('.'):
                response = response[:-1] + '!'
        
        # Mirror repetition if user uses it
        if style_analysis.get('uses_repetition'):
            # Add slight repetition to certain words
            response = re.sub(r'\bbanget\b', 'bangettt', response)
            response = re.sub(r'\basik\b', 'asikk', response)
        
        # Mirror emoji usage
        if style_analysis.get('uses_emoji') and self.user_personality_profile['emoji_usage']:
            if not re.search(r'[😀-🙏]', response):
                # Add appropriate emoji based on sentiment
                emoji_map = {
                    'happy': ' 😊',
                    'sadness': ' 😔',
                    'love': ' 🥰',
                    'anger': ' 😤',
                    'fear': ' 😰',
                    'neutral': ' 🙂'
                }
                sentiment = getattr(self, 'current_sentiment', 'neutral')
                response += emoji_map.get(sentiment, '')
        
        return response

    def update_short_term_memory(self, user_input: str, bot_response: str, sentiment: str):
        """Maintain short-term memory of last 3 exchanges"""
        self.short_term_memory.append({
            'user': user_input,
            'bot': bot_response,
            'sentiment': sentiment,
            'turn': self.turn_count
        })
        
        # Keep only last 3 messages
        if len(self.short_term_memory) > 3:
            self.short_term_memory.pop(0)

    def get_memory_context(self) -> str:
        """Generate context from short-term memory"""
        if not self.short_term_memory:
            return ""
        
        context_parts = []
        for memory in self.short_term_memory[-2:]:  # Last 2 for context
            context_parts.append(f"User pernah bilang: '{memory['user']}' (emosi: {memory['sentiment']})")
        
        return "Konteks percakapan sebelumnya: " + " | ".join(context_parts) if context_parts else ""

    def detect_transition(self, current_sentiment: str) -> Optional[str]:
        """Enhanced transition detection"""
        if self.previous_sentiment and self.previous_sentiment != current_sentiment:
            return f"{self.previous_sentiment} → {current_sentiment}"
        return None

    def create_adaptive_prompt(self, user_input: str, current_sentiment: str, transition: Optional[str], style_analysis: Dict) -> str:
        """Enhanced adaptive prompt with all new features"""
        empathy_level = self.get_empathy_level()
        
        # Get appropriate response style based on empathy progression
        if current_sentiment in self.response_styles:
            style_options = self.response_styles[current_sentiment][f'level_{empathy_level}']
        else:
            style_options = self.response_styles['neutral'][f'level_{empathy_level}']
        
        base_style = random.choice(style_options)
        
        # Memory context
        memory_context = self.get_memory_context()
        
        # Natural transition comment
        transition_comment = ""
        if transition:
            transition_comment = self.create_natural_transition_comment(transition)
        
        # Style mirroring instructions
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

        return prompt

    def chat(self, user_input: str) -> Dict:
        """Enhanced chat function with all improvements"""
        try:
            # Check for special cases first
            style_analysis = self.analyze_user_style(user_input)
            special_response = self.handle_special_cases(user_input, style_analysis)
            
            if special_response:
                mirrored_response = self.mirror_user_style(special_response, style_analysis)
                self.turn_count += 1
                return {
                    'response': mirrored_response,
                    'sentiment': 'neutral',
                    'confidence': 1.0,
                    'transition': None,
                    'special_case': True,
                    'style_analysis': style_analysis,
                    'empathy_level': self.get_empathy_level()
                }
            
            # Regular sentiment analysis with confidence check
            sentiment_result = self.analyze_sentiment(user_input)
            current_sentiment = sentiment_result['dominant_sentiment']
            self.current_sentiment = current_sentiment  # Store for emoji mirroring
            
            # Update personality profile
            self.update_personality_profile(style_analysis)
            
            # Detect emotional transition
            transition = self.detect_transition(current_sentiment)
            
            # Create adaptive prompt
            adaptive_prompt = self.create_adaptive_prompt(user_input, current_sentiment, transition, style_analysis)
            
            # Get response from Gemini
            response = self.chat_session.send_message(adaptive_prompt)
            response_text = response.text.strip()
            
            # Mirror user's communication style
            mirrored_response = self.mirror_user_style(response_text, style_analysis)
            
            # Update memory systems
            self.update_short_term_memory(user_input, mirrored_response, current_sentiment)
            self.previous_sentiment = current_sentiment
            self.turn_count += 1
            
            return {
                'response': mirrored_response,
                'sentiment': current_sentiment,
                'confidence': sentiment_result['confidence'],
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
        """Reset session for long conversations"""
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