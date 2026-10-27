import os
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import re

# Optional NLP imports
import importlib.util

SPACY_AVAILABLE = importlib.util.find_spec("spacy") is not None
if SPACY_AVAILABLE:
    import spacy
else:
    spacy = None

NLTK_AVAILABLE = importlib.util.find_spec("nltk") is not None
if NLTK_AVAILABLE:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
else:
    nltk = None
    stopwords = None
    word_tokenize = None

TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None
if TRANSFORMERS_AVAILABLE:
    from transformers import pipeline
else:
    pipeline = None

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
else:
    torch = None

GROQ_AVAILABLE = importlib.util.find_spec("groq") is not None
if GROQ_AVAILABLE:
    from groq import Groq
else:
    Groq = None

from database import get_db, ChatSession, ChatMessage
from schemas import ChatRequest, ChatResponse, ChatMessage as ChatMessageSchema

logger = logging.getLogger(__name__)

class MedicalChatbot:
    """Medical chatbot for liver cirrhosis assistance"""

    def __init__(self):
        # Initialize NLP components
        self.nlp = None
        self.sentiment_analyzer = None
        self.intent_classifier = None
        self.groq_client = None

        # Medical knowledge base
        self.medical_kb = self._load_medical_knowledge()

        # Conversation context
        self.context_window = 5  # Keep last 5 messages for context

        # Initialize components
        self._initialize_nlp_components()

    def _initialize_nlp_components(self):
        """Initialize NLP components"""
        try:
            # NLTK initialization
            if NLTK_AVAILABLE:
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)

                try:
                    nltk.data.find('corpora/stopwords')
                except LookupError:
                    nltk.download('stopwords', quiet=True)
            else:
                logger.warning("NLTK not installed - falling back to minimal text processing")

            # Load spaCy model
            if SPACY_AVAILABLE:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                    self.nlp = None
            else:
                logger.warning("spaCy not installed - using fallback tokenization")
                self.nlp = None

            # Initialize sentiment analyzer
            if TRANSFORMERS_AVAILABLE:
                try:
                    self.sentiment_analyzer = pipeline(
                        "sentiment-analysis",
                        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
                    )
                except Exception as e:
                    logger.warning(f"Sentiment analyzer failed to load: {str(e)}")
                    self.sentiment_analyzer = None
            else:
                logger.warning("Transformers not installed - skipping sentiment analysis")
                self.sentiment_analyzer = None

            # Initialize Groq client
            groq_api_key = os.getenv("GROQ_API_KEY")
            if GROQ_AVAILABLE and groq_api_key:
                self.groq_client = Groq(api_key=groq_api_key)
            else:
                if not GROQ_AVAILABLE:
                    logger.warning("Groq SDK not installed - chatbot will use rule-based responses.")
                else:
                    logger.warning("GROQ_API_KEY not found. Chatbot will use rule-based responses.")

        except Exception as e:
            logger.error(f"NLP initialization error: {str(e)}")

    def _load_medical_knowledge(self) -> Dict[str, Any]:
        """Load medical knowledge base"""
        return {
            "symptoms": {
                "jaundice": "Yellowing of skin and eyes due to bilirubin buildup",
                "ascites": "Abdominal fluid accumulation",
                "edema": "Swelling due to fluid retention",
                "fatigue": "Extreme tiredness and weakness",
                "weight_loss": "Unexplained weight loss",
                "abdominal_pain": "Pain in the abdominal area",
                "nausea": "Feeling of sickness with urge to vomit",
                "loss_of_appetite": "Reduced desire to eat"
            },
            "stages": {
                "stage_0": "Normal liver function",
                "stage_1": "Mild fibrosis - early scarring",
                "stage_2": "Moderate fibrosis - progressing scarring",
                "stage_3": "Severe fibrosis - advanced scarring",
                "stage_4": "Cirrhosis - severe liver damage"
            },
            "treatments": {
                "lifestyle": ["Avoid alcohol", "Maintain healthy diet", "Regular exercise", "Weight management"],
                "medications": ["Antiviral drugs for viral hepatitis", "Diuretics for fluid retention", "Beta-blockers for portal hypertension"],
                "procedures": ["Liver biopsy", "Endoscopic procedures", "Liver transplantation"],
                "monitoring": ["Regular blood tests", "Imaging studies", "Clinical examinations"]
            },
            "prevention": [
                "Vaccination against hepatitis A and B",
                "Safe sex practices",
                "Avoid sharing needles",
                "Limit alcohol consumption",
                "Maintain healthy weight",
                "Regular medical check-ups"
            ],
            "emergency_signs": [
                "Severe abdominal pain",
                "High fever",
                "Vomiting blood",
                "Black or bloody stools",
                "Confusion or drowsiness",
                "Difficulty breathing"
            ]
        }

    async def start_conversation(self, session_id: str, user_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        name = (user_info or {}).get("name", "there")
        return {
            "message": (
                f"Hello {name}. I can help collect liver-health symptoms, explain cirrhosis risk factors, "
                "and suggest when to consult a doctor. What would you like to discuss?"
            ),
            "next_question": "Are you experiencing fatigue, jaundice, abdominal swelling, or appetite loss?",
            "metadata": {"mode": "rule_based_medical_assistant"},
        }

    async def process_message(
        self,
        session_id: Optional[str] = None,
        message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        request: Optional[ChatRequest] = None,
    ) -> Dict[str, Any]:
        """Process user message and generate response"""
        try:
            if request is not None:
                message = getattr(request, "message", message)
                session_id = getattr(request, "session_id", session_id)

            message = message or ""
            context_messages = []

            analysis = self._analyze_message(message)
            # Determine intent
            intent = self._classify_intent(message, analysis)

            # Generate response based on intent
            response_text = await self._generate_response(intent, analysis, context_messages)

            return {
                "message": response_text,
                "next_question": self._next_question_for_intent(intent),
                "completed": False,
                "metadata": {
                    "intent": intent,
                    "confidence": analysis.get("confidence", 0.5),
                    "suggestions": self._generate_suggestions(intent),
                    "emergency_flags": self._check_emergency_flags(message),
                },
            }

        except Exception as e:
            logger.error(f"Message processing error: {str(e)}")
            return {
                "message": "I could not process that safely. Please try again or consult a healthcare professional.",
                "next_question": None,
                "completed": False,
                "metadata": {"intent": "error"},
            }

    def _next_question_for_intent(self, intent: str) -> Optional[str]:
        questions = {
            "symptom_inquiry": "How long have these symptoms been present?",
            "diagnosis_inquiry": "Do you have any recent ultrasound, CT, MRI, or liver test results?",
            "treatment_inquiry": "Has a doctor already diagnosed a liver condition?",
            "prevention_inquiry": "Would you like diet, alcohol, vaccination, or exercise guidance?",
            "emergency": "Are any emergency symptoms happening right now?",
        }
        return questions.get(intent)

    def _analyze_message(self, message: str) -> Dict[str, Any]:
        """Analyze user message for intent and entities"""
        analysis = {
            'tokens': [],
            'entities': [],
            'sentiment': 'neutral',
            'confidence': 0.5,
            'keywords': []
        }

        try:
            # Tokenize and clean
            if self.nlp:
                doc = self.nlp(message.lower())
                analysis['tokens'] = [token.text for token in doc if not token.is_stop and token.is_alpha]
                analysis['entities'] = [(ent.text, ent.label_) for ent in doc.ents]
            elif NLTK_AVAILABLE and word_tokenize is not None and stopwords is not None:
                tokens = word_tokenize(message.lower())
                stop_words = set(stopwords.words('english'))
                analysis['tokens'] = [t for t in tokens if t not in stop_words and t.isalpha()]
            else:
                # Minimal fallback tokenization
                tokens = re.findall(r"\b[a-zA-Z]+\b", message.lower())
                analysis['tokens'] = [t for t in tokens if len(t) > 1]

            # Extract keywords
            analysis['keywords'] = self._extract_keywords(message)

            # Sentiment analysis
            if self.sentiment_analyzer:
                try:
                    sentiment_result = self.sentiment_analyzer(message)[0]
                    analysis['sentiment'] = sentiment_result['label'].lower()
                    analysis['confidence'] = sentiment_result['score']
                except Exception as e:
                    logger.warning(f"Sentiment analysis failed: {str(e)}")

        except Exception as e:
            logger.error(f"Message analysis error: {str(e)}")

        return analysis

    def _classify_intent(self, message: str, analysis: Dict[str, Any]) -> str:
        """Classify user intent"""
        message_lower = message.lower()
        keywords = analysis.get('keywords', [])

        # Symptom-related queries
        symptom_keywords = ['symptom', 'symptoms', 'pain', 'fatigue', 'jaundice', 'swelling', 'nausea', 'vomiting']
        if any(kw in message_lower for kw in symptom_keywords):
            return 'symptom_inquiry'

        # Treatment queries
        treatment_keywords = ['treatment', 'cure', 'medicine', 'medication', 'therapy', 'surgery']
        if any(kw in message_lower for kw in treatment_keywords):
            return 'treatment_inquiry'

        # Diagnosis queries
        diagnosis_keywords = ['diagnosis', 'stage', 'condition', 'disease', 'cirrhosis', 'fibrosis']
        if any(kw in message_lower for kw in diagnosis_keywords):
            return 'diagnosis_inquiry'

        # Prevention queries
        prevention_keywords = ['prevent', 'prevention', 'avoid', 'risk', 'cause']
        if any(kw in message_lower for kw in prevention_keywords):
            return 'prevention_inquiry'

        # Emergency queries
        emergency_keywords = ['emergency', 'urgent', 'bleeding', 'blood', 'severe', 'help']
        if any(kw in message_lower for kw in emergency_keywords):
            return 'emergency'

        # General questions
        if any(word in message_lower for word in ['what', 'how', 'why', 'when', 'where', 'can', 'should']):
            return 'general_question'

        # Greeting
        greeting_keywords = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(kw in message_lower for kw in greeting_keywords):
            return 'greeting'

        # Default
        return 'general_conversation'

    async def _generate_response(self, intent: str, analysis: Dict[str, Any], context: List[Dict]) -> str:
        """Generate response based on intent"""
        try:
            # Try Groq API first for complex responses
            if self.groq_client and intent in ['general_question', 'diagnosis_inquiry', 'treatment_inquiry']:
                return await self._generate_groq_response(intent, analysis, context)

            # Use rule-based responses
            return self._generate_rule_based_response(intent, analysis)

        except Exception as e:
            logger.error(f"Response generation error: {str(e)}")
            return self._generate_rule_based_response(intent, analysis)

    async def _generate_groq_response(self, intent: str, analysis: Dict[str, Any], context: List[Dict]) -> str:
        """Generate response using Groq API"""
        try:
            # Build context string
            context_str = ""
            if context:
                context_str = "Previous conversation:\n"
                for msg in context[-3:]:  # Last 3 messages
                    context_str += f"User: {msg['user_message']}\nAssistant: {msg['bot_response']}\n"

            # Create prompt
            system_prompt = """You are a medical assistant specializing in liver cirrhosis.
            Provide accurate, helpful information about liver health, cirrhosis symptoms, treatments, and prevention.
            Always recommend consulting healthcare professionals for medical advice.
            Be empathetic and supportive."""

            user_prompt = f"""
            Context: {context_str}
            User message: {analysis.get('original_message', '')}
            Intent: {intent}
            Keywords: {', '.join(analysis.get('keywords', []))}

            Provide a helpful response about liver cirrhosis.
            """

            response = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            return self._generate_rule_based_response(intent, analysis)

    def _generate_rule_based_response(self, intent: str, analysis: Dict[str, Any]) -> str:
        """Generate rule-based response"""
        responses = {
            'greeting': "Hello! I'm here to help you with information about liver cirrhosis. How can I assist you today?",

            'symptom_inquiry': f"I understand you're asking about symptoms. Common symptoms of liver cirrhosis include: {', '.join(self.medical_kb['symptoms'].keys())}. Please remember that only a healthcare professional can provide a proper diagnosis.",

            'treatment_inquiry': f"For liver cirrhosis treatment, options include lifestyle changes, medications, and in severe cases, procedures like liver transplantation. The specific treatment depends on the stage and underlying cause. Please consult a hepatologist for personalized advice.",

            'diagnosis_inquiry': f"Liver cirrhosis is diagnosed through blood tests, imaging studies, and sometimes liver biopsy. The condition progresses through stages from mild fibrosis to advanced cirrhosis. Early detection is crucial for better outcomes.",

            'prevention_inquiry': f"To prevent liver cirrhosis: {'. '.join(self.medical_kb['prevention'])}. Regular medical check-ups and maintaining a healthy lifestyle are essential.",

            'emergency': "If you're experiencing severe symptoms like vomiting blood, black stools, confusion, or difficulty breathing, please seek immediate medical attention or call emergency services. This is a medical emergency!",

            'general_question': "I'd be happy to help with your question about liver cirrhosis. Could you please provide more details about what you'd like to know?",

            'general_conversation': "I'm here to provide information about liver cirrhosis. Feel free to ask about symptoms, treatments, prevention, or any other related topics."
        }

        return responses.get(intent, responses['general_conversation'])

    def _extract_keywords(self, message: str) -> List[str]:
        """Extract medical keywords from message"""
        medical_terms = [
            'liver', 'cirrhosis', 'fibrosis', 'jaundice', 'ascites', 'edema', 'fatigue',
            'alcohol', 'hepatitis', 'transplant', 'biopsy', 'ultrasound', 'ct', 'mri',
            'bilirubin', 'albumin', 'ast', 'alt', 'alkaline', 'phosphatase', 'platelets',
            'prothrombin', 'copper', 'cholesterol', 'triglycerides'
        ]

        message_lower = message.lower()
        keywords = []

        for term in medical_terms:
            if term in message_lower:
                keywords.append(term)

        return keywords

    def _generate_suggestions(self, intent: str) -> List[str]:
        """Generate follow-up suggestions"""
        suggestions = {
            'symptom_inquiry': [
                "Describe your symptoms in detail",
                "Ask about when to see a doctor",
                "Learn about symptom management"
            ],
            'treatment_inquiry': [
                "Ask about treatment options",
                "Learn about lifestyle changes",
                "Understand medication side effects"
            ],
            'diagnosis_inquiry': [
                "Ask about diagnostic tests",
                "Learn about disease stages",
                "Understand prognosis"
            ],
            'prevention_inquiry': [
                "Learn about risk factors",
                "Ask about screening tests",
                "Understand healthy lifestyle choices"
            ],
            'emergency': [
                "Call emergency services immediately",
                "Go to nearest hospital",
                "Contact your healthcare provider"
            ]
        }

        return suggestions.get(intent, ["Ask another question", "Learn more about liver health"])

    def _check_emergency_flags(self, message: str) -> List[str]:
        """Check for emergency indicators"""
        emergency_indicators = [
            'vomiting blood', 'blood in stool', 'black stool', 'severe pain',
            'confusion', 'drowsiness', 'difficulty breathing', 'high fever',
            'unconscious', 'chest pain', 'severe headache'
        ]

        message_lower = message.lower()
        flags = []

        for indicator in emergency_indicators:
            if indicator in message_lower:
                flags.append(indicator)

        return flags

    async def _get_conversation_context(self, session_id: str) -> List[Dict]:
        """Get recent conversation context"""
        try:
            db = next(get_db())
            messages = db.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).order_by(ChatMessage.created_at.desc()).limit(self.context_window).all()

            context = []
            for msg in reversed(messages):  # Oldest first
                context.append({
                    'user_message': msg.user_message,
                    'bot_response': msg.bot_response,
                    'intent': msg.intent,
                    'timestamp': msg.created_at
                })

            return context

        except Exception as e:
            logger.error(f"Context retrieval error: {str(e)}")
            return []

    async def _save_message(self, session_id: str, user_message: str, bot_response: str, intent: str):
        """Save conversation message to database"""
        try:
            db = next(get_db())

            # Create or get session
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if not session:
                session = ChatSession(id=session_id)
                db.add(session)
                db.commit()

            # Save message
            chat_message = ChatMessage(
                session_id=session_id,
                user_message=user_message,
                bot_response=bot_response,
                intent=intent
            )

            db.add(chat_message)
            db.commit()

        except Exception as e:
            logger.error(f"Message save error: {str(e)}")

# Global chatbot instance
chatbot = MedicalChatbot()
