"""
RAG-Based Medical Chatbot
Uses LangChain + Retrieval-Augmented Generation for medical Q&A
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LangChain imports
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available. Install with: pip install langchain")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class MedicalChatbot:
    """RAG-based chatbot for medical Q&A on liver health."""
    
    MEDICAL_KNOWLEDGE_BASE = """
    LIVER CIRRHOSIS STAGING AND CLASSIFICATION:
    
    F0 - Normal Liver: No fibrosis detected. Liver function is normal.
    F1 - Mild Fibrosis: Portal fibrosis without septa. Early scarring.
    F2 - Moderate Fibrosis: Portal fibrosis with occasional septa. Progressive scarring.
    F3 - Severe Fibrosis: Numerous septa without cirrhosis. Advanced stage.
    F4 - Cirrhosis: Complete cirrhotic architecture. Liver failure risk is high.
    
    SYMPTOMS OF LIVER DISEASE:
    - Jaundice (yellowing of skin/eyes)
    - Abdominal swelling (ascites)
    - Fatigue and weakness
    - Nausea and loss of appetite
    - Dark urine or pale stools
    - Itching
    - Leg swelling (edema)
    - Cognitive changes (hepatic encephalopathy)
    
    RISK FACTORS:
    - Excessive alcohol consumption
    - Viral hepatitis (B, C)
    - Non-alcoholic fatty liver disease (NAFLD)
    - Autoimmune liver disease
    - Hemochromatosis
    - Wilson's disease
    - Obesity
    - Diabetes
    
    DIAGNOSTIC METHODS:
    - Blood tests: Liver function tests, bilirubin, albumin, platelets
    - Imaging: Ultrasound, CT, MRI
    - Biopsy: Gold standard for fibrosis staging
    - Non-invasive: FibroScan (transient elastography)
    - AI-assisted: EfficientNet imaging classification
    
    MANAGEMENT STRATEGIES:
    - F0-F1: Lifestyle modification, treat underlying cause
    - F2-F3: Medical treatment, antiviral therapy, monitoring
    - F4: Comprehensive cirrhosis management, hepatology referral, transplant evaluation
    
    PREVENTION:
    - Limit alcohol consumption
    - Maintain healthy weight
    - Regular vaccination against Hepatitis A & B
    - Safe practices to prevent Hepatitis C transmission
    - Regular health check-ups
    - Manage chronic diseases (diabetes, obesity)
    
    PROGNOSIS:
    - Early stages (F0-F1) are potentially reversible with treatment
    - Later stages (F3-F4) require intensive management
    - Early detection improves outcomes significantly
    - Regular monitoring is essential for disease progression assessment
    """
    
    def __init__(self, api_key: Optional[str] = None, llm_provider: str = 'openai'):
        """
        Initialize chatbot.
        
        Args:
            api_key: API key for LLM service
            llm_provider: 'openai', 'groq', or 'local'
        """
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.retriever = None
        self.qa_chain = None
        self.conversation_history = []
        self.vector_store = None
        
        self._initialize_rag()
    
    def _initialize_rag(self):
        """Initialize RAG system."""
        try:
            logger.info(f"Initializing chatbot with {self.llm_provider} provider...")
            
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Create vector store from knowledge base
            self._build_knowledge_base()
            
            # Initialize LLM
            self._initialize_llm()
            
            logger.info("Chatbot initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing chatbot: {e}")
    
    def _build_knowledge_base(self):
        """Build vector store from medical knowledge base."""
        try:
            logger.info("Building knowledge base...")
            
            # Split text into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = splitter.split_text(self.MEDICAL_KNOWLEDGE_BASE)
            logger.info(f"Created {len(chunks)} knowledge base chunks")
            
            # Create vector store
            self.vector_store = Chroma.from_texts(
                texts=chunks,
                embedding=self.embeddings,
                collection_name="medical_liver"
            )
            
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 3}
            )
            
            logger.info("Knowledge base built successfully")
            
        except Exception as e:
            logger.error(f"Error building knowledge base: {e}")
    
    def _initialize_llm(self):
        """Initialize LLM based on provider."""
        try:
            if self.llm_provider == 'openai':
                self._initialize_openai()
            elif self.llm_provider == 'groq':
                self._initialize_groq()
            else:
                self._initialize_local()
                
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
    
    def _initialize_openai(self):
        """Initialize OpenAI LLM."""
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI not available. Falling back to local.")
            self._initialize_local()
            return
        
        if self.api_key:
            openai.api_key = self.api_key
        
        try:
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=500
            )
            
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=self._get_prompt_template()
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True
            )
            
            logger.info("OpenAI LLM initialized")
            
        except Exception as e:
            logger.error(f"Error with OpenAI: {e}")
    
    def _initialize_groq(self):
        """Initialize Groq LLM (faster, free alternative)."""
        if not GROQ_AVAILABLE:
            logger.warning("Groq not available. Falling back to local.")
            self._initialize_local()
            return
        
        self.groq_client = Groq(api_key=self.api_key)
        logger.info("Groq LLM initialized")
    
    def _initialize_local(self):
        """Initialize local LLM (fallback)."""
        logger.info("Using fallback rule-based response system")
        self.qa_chain = None
    
    def _get_prompt_template(self) -> str:
        """Get system prompt for medical chatbot."""
        return """You are a professional medical assistant specialized in hepatology and liver health. 
        You provide accurate, evidence-based information about liver diseases, cirrhosis staging, and general guidance.
        
        IMPORTANT GUIDELINES:
        1. Always provide disclaimers that you're not a substitute for professional medical advice
        2. Encourage users to consult qualified healthcare professionals
        3. Be clear about what is general information vs. specific medical advice
        4. Use medical terminology accurately but explain in simple terms
        5. Do not diagnose or prescribe medications
        
        Context from medical knowledge base:
        {context}
        
        User Question: {question}
        
        Provide a helpful, accurate response based on the context above."""
    
    def chat(self, user_message: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get response from chatbot.
        
        Args:
            user_message: User's question or message
            user_id: Optional user ID for conversation tracking
        
        Returns:
            Dictionary with response and metadata
        """
        logger.info(f"Processing user message: {user_message[:50]}...")
        
        try:
            # Store conversation
            self.conversation_history.append({
                'role': 'user',
                'content': user_message,
                'timestamp': datetime.now().isoformat()
            })
            
            # Get response
            if self.llm_provider == 'groq':
                response = self._groq_response(user_message)
            elif self.qa_chain:
                response = self._rag_response(user_message)
            else:
                response = self._rule_based_response(user_message)
            
            # Store response
            self.conversation_history.append({
                'role': 'assistant',
                'content': response['answer'],
                'timestamp': datetime.now().isoformat()
            })
            
            return {
                'response': response['answer'],
                'sources': response.get('sources', []),
                'confidence': response.get('confidence', 'medium'),
                'conversation_id': user_id
            }
            
        except Exception as e:
            logger.error(f"Error processing chat: {e}")
            return {
                'response': f"I apologize, but I encountered an error: {str(e)}. Please try again.",
                'error': str(e),
                'conversation_id': user_id
            }
    
    def _rag_response(self, user_message: str) -> Dict[str, Any]:
        """Get response using RAG chain."""
        try:
            result = self.qa_chain({
                "query": user_message
            })
            
            sources = [doc.metadata for doc in result.get('source_documents', [])]
            
            return {
                'answer': result['result'],
                'sources': sources,
                'confidence': 'high'
            }
            
        except Exception as e:
            logger.error(f"Error in RAG response: {e}")
            return self._rule_based_response(user_message)
    
    def _groq_response(self, user_message: str) -> Dict[str, Any]:
        """Get response from Groq API."""
        try:
            # Retrieve relevant context
            relevant_docs = self.retriever.get_relevant_documents(user_message)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            # Get response from Groq
            message = self.groq_client.messages.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"""You are a medical assistant specializing in liver health.
                        
Medical Context:
{context}

User Question: {user_message}

Provide a helpful, accurate response based on the context above. 
Remember to include appropriate disclaimers about consulting healthcare professionals."""
                    }
                ],
                model="mixtral-8x7b-32768",
            )
            
            return {
                'answer': message.content[0].text,
                'sources': [],
                'confidence': 'high'
            }
            
        except Exception as e:
            logger.error(f"Error with Groq: {e}")
            return self._rule_based_response(user_message)
    
    def _rule_based_response(self, user_message: str) -> Dict[str, Any]:
        """Fallback rule-based response system."""
        message_lower = user_message.lower()
        
        responses = {
            'stage': "Liver cirrhosis is classified into 5 stages (F0-F4): F0=Normal, F1=Mild fibrosis, F2=Moderate, F3=Severe, F4=Cirrhosis. Each stage requires different management approaches.",
            'symptom': "Common symptoms include jaundice, fatigue, abdominal swelling, nausea, and dark urine. If you experience these, please consult a healthcare professional.",
            'risk': "Risk factors include excessive alcohol, hepatitis B/C, obesity, and diabetes. Limiting alcohol and maintaining health can reduce risk.",
            'treatment': "Treatment depends on the stage and cause. Early stages may be managed with lifestyle changes, while advanced stages require medical intervention.",
            'diagnosis': "Diagnosis involves blood tests, imaging (ultrasound/CT/MRI), and sometimes biopsy. Our AI system can assist with non-invasive image analysis.",
        }
        
        for keyword, response_text in responses.items():
            if keyword in message_lower:
                return {
                    'answer': response_text + "\n\nPlease consult a healthcare professional for personalized medical advice.",
                    'sources': [],
                    'confidence': 'medium'
                }
        
        # Default response
        return {
            'answer': "I can provide information about liver health, cirrhosis staging, symptoms, and general medical guidance. Please ask me about liver-related topics. For specific medical advice, please consult a healthcare professional.",
            'sources': [],
            'confidence': 'low'
        }
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history."""
        return self.conversation_history.copy()
    
    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def save_conversation(self, file_path: str):
        """Save conversation to file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
            logger.info(f"Conversation saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")


def create_chatbot(api_key: Optional[str] = None,
                  provider: str = 'openai') -> MedicalChatbot:
    """Factory function to create medical chatbot."""
    return MedicalChatbot(api_key=api_key, llm_provider=provider)


if __name__ == "__main__":
    # Example usage
    chatbot = create_chatbot(provider='local')
    
    # Test conversation
    response = chatbot.chat("What are the stages of liver cirrhosis?")
    print("User: What are the stages of liver cirrhosis?")
    print(f"Bot: {response['response']}\n")
    
    response = chatbot.chat("What symptoms should I watch for?")
    print("User: What symptoms should I watch for?")
    print(f"Bot: {response['response']}")
