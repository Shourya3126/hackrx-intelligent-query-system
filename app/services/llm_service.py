import openai
from typing import List, Dict, Any
from app.config import settings
import logging


logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        # OpenRouter client using OpenAI SDK
        self.client = openai.OpenAI(
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY,
            default_headers={
                "HTTP-Referer": "https://hackrx-app.com",  # Optional
                "X-Title": "HackRx Document QA System"  # Optional
            }
        )
        logger.info(f"Initialized OpenRouter with model: {settings.OPENROUTER_MODEL}")
    
    async def generate_answers(self, questions: List[str], contexts: List[List[Dict[str, Any]]]) -> List[str]:
        """Generate answers for questions using retrieved contexts"""
        answers = []
        
        for question, context_chunks in zip(questions, contexts):
            try:
                answer = await self._generate_single_answer(question, context_chunks)
                answers.append(answer)
            except Exception as e:
                logger.error(f"Error generating answer for question '{question}': {str(e)}")
                # Provide a more helpful error message instead of generic response
                error_msg = f"Unable to process this question due to API limitations. Please try again."
                answers.append(error_msg)
        
        return answers
    
    async def _generate_single_answer(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate answer for a single question using OpenRouter"""
        
        # Check if we have context
        if not context_chunks:
            return "No relevant information found in the document to answer this question."
        
        # Prepare context from retrieved chunks
        context_text = "\n\n".join([
            f"[Document Section {i+1}]:\n{chunk['text']}"
            for i, chunk in enumerate(context_chunks[:settings.MAX_CHUNKS_FOR_CONTEXT])
        ])
        
        # Enhanced prompt for better insurance document analysis
        prompt = f"""You are an expert insurance policy analyst. Based on the provided policy document sections, answer the user's question accurately and concisely.

POLICY DOCUMENT CONTEXT:
{context_text}

QUESTION: {question}

INSTRUCTIONS:
- Provide a direct, accurate answer based solely on the policy context above
- Quote specific policy language when relevant (use quotes "like this")
- If the information isn't clearly stated in the context, say so honestly
- Be concise but complete in your response
- Focus on key details like amounts, timeframes, conditions, and limitations
- If there are multiple conditions or requirements, list them clearly

ANSWER:"""

        try:
            response = self.client.chat.completions.create(
                model=settings.OPENROUTER_MODEL,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a precise insurance policy expert who provides accurate, evidence-based answers. Always base your responses on the provided document context."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE,
                timeout=20  # Add timeout to prevent hanging
            )
            
            # Robust response handling to fix NoneType errors
            if response and hasattr(response, 'choices') and response.choices:
                if len(response.choices) > 0 and response.choices[0].message:
                    content = response.choices[0].message.content
                    if content and content.strip():
                        answer = content.strip()
                        logger.info(f"Successfully generated answer for question: {question[:50]}...")
                        return answer
                    else:
                        logger.warning(f"Empty content returned for question: {question[:50]}...")
                        return "The model returned an empty response. This might be due to content filtering or processing issues."
                else:
                    logger.warning(f"No valid choices in response for question: {question[:50]}...")
                    return "The model did not return a valid response structure."
            else:
                logger.warning(f"Invalid response structure for question: {question[:50]}...")
                return "Received an invalid response from the language model."
                
        except openai.APITimeoutError:
            logger.error(f"Timeout error for question: {question[:50]}...")
            return "The request timed out while processing this question. Please try again."
            
        except openai.RateLimitError:
            logger.error(f"Rate limit error for question: {question[:50]}...")
            return "Rate limit exceeded. Please try again in a moment."
            
        except openai.APIError as e:
            logger.error(f"OpenRouter API error for question '{question[:50]}...': {str(e)}")
            return f"API error occurred while processing this question: {str(e)[:100]}..."
            
        except Exception as e:
            logger.error(f"Unexpected error for question '{question[:50]}...': {str(e)}")
            return f"An unexpected error occurred: {str(e)[:100]}..."

    def _validate_response(self, response) -> bool:
        """Validate OpenRouter response structure"""
        try:
            return (
                response and 
                hasattr(response, 'choices') and 
                response.choices and 
                len(response.choices) > 0 and 
                response.choices[0].message and 
                response.choices[0].message.content
            )
        except Exception:
            return False
