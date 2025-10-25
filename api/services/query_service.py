"""
Query Service - Business logic for document querying with LLM
Multi-provider support: Ollama, OpenAI, OpenRouter, Anthropic
"""

import logging
import asyncio
from typing import List, Dict, Optional
import httpx
import base64

from colivara_py import ColiVara

from api.core.config import settings
from api.processors.query_processor import expand_query_terms, is_relevant_section

logger = logging.getLogger(__name__)


class QueryService:
    """Service for processing document queries with multi-provider LLM support"""
    
    def __init__(self):
        self.ollama_url = settings.OLLAMA_URL
        self.model_name = settings.MODEL_NAME
        self.colivara_base_url = settings.COLIVARA_BASE_URL_QUERY
        self.supported_providers = ['ollama', 'openai', 'openrouter', 'anthropic']
    
    async def process_query(
        self,
        query: str,
        api_key: str,
        collection_name: str,
        include_next_page: bool = False,
        max_additional_pages: int = 1,
        allow_general_fallback: bool = True,
        similarity_threshold: float = 0.25,
        provider: str = 'ollama',
        provider_settings: Dict = None
    ) -> Dict:
        """
        Process a query against documents with multi-provider support
        
        Args:
            query: User's question
            api_key: API key for ColiVara authentication
            collection_name: Collection to search
            include_next_page: Include consecutive pages
            max_additional_pages: Max pages to include
            allow_general_fallback: Use general knowledge if no docs found
            similarity_threshold: Minimum similarity score
            provider: LLM provider ('ollama', 'openai', 'openrouter', 'anthropic')
            provider_settings: Provider-specific settings (API keys, models, etc.)
            
        Returns:
            Query response dictionary
        """
        provider_settings = provider_settings or {}
        debug_info = []
        
        try:
            debug_info.append(f"Starting query: {query}")
            debug_info.append(f"Similarity threshold: {similarity_threshold}")
            debug_info.append(f"Collection: {collection_name}")
            
            # Step 1: Expand query
            expanded_query = expand_query_terms(query)
            debug_info.append(f"Expanded query: {expanded_query}")
            
            # Step 2: Search ColiVara
            rag_client = ColiVara(api_key=api_key, base_url=self.colivara_base_url)
            
            search_top_k = 15
            
            results = await asyncio.to_thread(
                rag_client.search,
                query=expanded_query,
                collection_name=collection_name,
                top_k=search_top_k
            )
            
            # Check if we have results
            if not results.results:
                debug_info.append("No documents found in search results")
                if allow_general_fallback:
                    debug_info.append("Using general knowledge fallback")
                    return await self.general_knowledge_fallback(query, debug_info)
                else:
                    return {
                        "result": "No matching documents found",
                        "source_info": {},
                        "debug_info": debug_info,
                        "images_info": []
                    }
            
            # Filter by threshold
            filtered_results = [
                doc for doc in results.results
                if doc.normalized_score >= similarity_threshold
            ]
            
            if not filtered_results:
                debug_info.append(f"No documents meet similarity threshold ({similarity_threshold})")
                if allow_general_fallback:
                    debug_info.append("Using general knowledge fallback")
                    return await self.general_knowledge_fallback(query, debug_info)
                else:
                    return {
                        "result": "No documents meet similarity threshold",
                        "source_info": {},
                        "debug_info": debug_info,
                        "images_info": []
                    }
            
            # Boost relevant sections
            for doc in filtered_results:
                doc_text = getattr(doc, 'text', '') or getattr(doc, 'content', '')
                if is_relevant_section(doc_text, query):
                    doc.normalized_score = min(doc.normalized_score * 1.5, 1.0)
                    debug_info.append(
                        f"Boosted document: {getattr(doc, 'document_name', '')} "
                        f"(page {getattr(doc, 'page_number', '?')})"
                    )
            
            # Re-sort by boosted score
            filtered_results = sorted(
                filtered_results,
                key=lambda x: x.normalized_score,
                reverse=True
            )
            
            # Get document pages
            max_docs = 3
            page_docs = self._get_document_pages(filtered_results, max_docs)
            debug_info.append(f"Using {len(page_docs)} document pages")
            
            # Prepare images for VLM
            images_data = self._combine_images_for_vlm(page_docs)
            
            # Generate response using LLM
            if images_data:
                llm_response = await self._query_with_llm(query, images_data, provider, provider_settings)
            else:
                # Text-only fallback
                llm_response = await self._query_text_only(query, page_docs, provider, provider_settings)
            
            # Prepare source info
            source_info = self._prepare_source_info(page_docs)
            
            return {
                "result": llm_response,
                "source_info": source_info,
                "debug_info": debug_info,
                "images_info": [
                    {
                        "document_name": img.get('document_name', ''),
                        "page_number": img.get('page_number', 0),
                        "base64_preview": img.get('base64', '')[:100] if img.get('base64') else None
                    }
                    for img in images_data
                ]
            }
        
        except Exception as e:
            logger.error(f"Query processing error: {str(e)}", exc_info=True)
            debug_info.append(f"Error: {str(e)}")
            return {
                "result": f"Error processing query: {str(e)}",
                "source_info": {},
                "debug_info": debug_info,
                "images_info": []
            }
    
    def _get_document_pages(self, filtered_results: List, max_docs: int) -> List:
        """Get pages from top documents"""
        page_docs = []
        seen_docs = set()
        
        for doc in filtered_results:
            doc_name = getattr(doc, 'document_name', '')
            if doc_name not in seen_docs and len(page_docs) < max_docs:
                page_docs.append(doc)
                seen_docs.add(doc_name)
        
        return page_docs
    
    def _combine_images_for_vlm(self, page_docs: List) -> List[Dict]:
        """Prepare images for VLM processing"""
        combined_images = []
        
        for doc in page_docs:
            if not hasattr(doc, 'img_base64') or not doc.img_base64:
                continue
            
            base64_data = doc.img_base64
            if base64_data.startswith('data:'):
                base64_data = base64_data.split(',', 1)[1]
            
            combined_images.append({
                'base64': base64_data,
                'page_number': getattr(doc, 'page_number', 'unknown'),
                'document_name': getattr(doc, 'document_name', 'unknown'),
                'score': getattr(doc, 'normalized_score', 0)
            })
        
        return combined_images
    
    def _prepare_source_info(self, page_docs: List) -> Dict:
        """Prepare source information"""
        sources = []
        
        for doc in page_docs:
            sources.append({
                "document": getattr(doc, 'document_name', 'unknown'),
                "page": getattr(doc, 'page_number', 'unknown'),
                "score": getattr(doc, 'normalized_score', 0)
            })
        
        return {
            "sources": sources,
            "total_documents": len(set(s["document"] for s in sources))
        }
    
    async def _query_with_llm(self, query: str, images_data: List[Dict], provider: str = 'ollama', provider_settings: Dict = None) -> str:
        """Query LLM with image data - multi-provider support"""
        provider_settings = provider_settings or {}
        
        try:
            if provider == 'ollama':
                return await self._query_ollama_vision(query, images_data, provider_settings)
            elif provider == 'openai':
                return await self._query_openai_vision(query, images_data, provider_settings)
            elif provider == 'openrouter':
                return await self._query_openrouter_vision(query, images_data, provider_settings)
            elif provider == 'anthropic':
                return await self._query_anthropic_vision(query, images_data, provider_settings)
            else:
                logger.warning(f"Unknown provider: {provider}, falling back to Ollama")
                return await self._query_ollama_vision(query, images_data, provider_settings)
        
        except Exception as e:
            logger.error(f"LLM query error with provider {provider}: {str(e)}")
            return f"Error querying LLM ({provider}): {str(e)}"
    
    async def _query_ollama_vision(self, query: str, images_data: List[Dict], settings_dict: Dict) -> str:
        """Query Ollama with vision support"""
        try:
            ollama_url = settings_dict.get('url', self.ollama_url)
            model = settings_dict.get('model', self.model_name)
            
            messages = [{
                "role": "user",
                "content": query,
                "images": [img['base64'] for img in images_data if img.get('base64')]
            }]
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{ollama_url}/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('message', {}).get('content', 'No response from LLM')
                else:
                    logger.error(f"Ollama error: {response.status_code} - {response.text}")
                    return f"Error querying Ollama: {response.status_code}"
        
        except Exception as e:
            logger.error(f"Ollama query error: {str(e)}")
            return f"Error querying Ollama: {str(e)}"
    
    async def _query_openai_vision(self, query: str, images_data: List[Dict], settings_dict: Dict) -> str:
        """Query OpenAI GPT-4 Vision"""
        try:
            api_key = settings_dict.get('apiKey', '')
            model = settings_dict.get('model', 'gpt-4o')
            
            if not api_key:
                return "Error: OpenAI API key not configured"
            
            # Build content with images
            content = [{"type": "text", "text": query}]
            for img in images_data:
                if img.get('base64'):
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img['base64']}"
                        }
                    })
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": content
                            }
                        ],
                        "max_tokens": 2000
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                else:
                    logger.error(f"OpenAI error: {response.status_code} - {response.text}")
                    return f"Error querying OpenAI: {response.status_code} - {response.text}"
        
        except Exception as e:
            logger.error(f"OpenAI query error: {str(e)}")
            return f"Error querying OpenAI: {str(e)}"
    
    async def _query_openrouter_vision(self, query: str, images_data: List[Dict], settings_dict: Dict) -> str:
        """Query OpenRouter with vision models"""
        try:
            api_key = settings_dict.get('apiKey', '')
            model = settings_dict.get('model', 'google/gemini-pro-vision')
            
            if not api_key:
                return "Error: OpenRouter API key not configured"
            
            # Build content with images (OpenRouter uses OpenAI-compatible format)
            content = [{"type": "text", "text": query}]
            for img in images_data:
                if img.get('base64'):
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img['base64']}"
                        }
                    })
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/yourusername/document-qa",
                        "X-Title": "Document Q&A"
                    },
                    json={
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": content
                            }
                        ]
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                else:
                    logger.error(f"OpenRouter error: {response.status_code} - {response.text}")
                    return f"Error querying OpenRouter: {response.status_code} - {response.text}"
        
        except Exception as e:
            logger.error(f"OpenRouter query error: {str(e)}")
            return f"Error querying OpenRouter: {str(e)}"
    
    async def _query_anthropic_vision(self, query: str, images_data: List[Dict], settings_dict: Dict) -> str:
        """Query Anthropic Claude with vision support"""
        try:
            api_key = settings_dict.get('apiKey', '')
            model = settings_dict.get('model', 'claude-3-opus-20240229')
            
            if not api_key:
                return "Error: Anthropic API key not configured"
            
            # Build content with images (Anthropic format)
            content = []
            for img in images_data:
                if img.get('base64'):
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img['base64']
                        }
                    })
            
            content.append({
                "type": "text",
                "text": query
            })
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model,
                        "max_tokens": 2000,
                        "messages": [
                            {
                                "role": "user",
                                "content": content
                            }
                        ]
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['content'][0]['text']
                else:
                    logger.error(f"Anthropic error: {response.status_code} - {response.text}")
                    return f"Error querying Anthropic: {response.status_code} - {response.text}"
        
        except Exception as e:
            logger.error(f"Anthropic query error: {str(e)}")
            return f"Error querying Anthropic: {str(e)}"
    
    async def _query_text_only(self, query: str, page_docs: List, provider: str = 'ollama', provider_settings: Dict = None) -> str:
        """Query with text-only (no images) - multi-provider support"""
        provider_settings = provider_settings or {}
        
        try:
            # Extract text from documents
            context_parts = []
            for doc in page_docs:
                doc_text = getattr(doc, 'text', '') or getattr(doc, 'content', '')
                if doc_text:
                    context_parts.append(doc_text)
            
            context = "\n\n".join(context_parts[:3])  # Limit to 3 documents
            
            # Build prompt
            prompt_text = f"""Based on the following document context, please answer the question.

Context:
{context}

Question: {query}

Answer:"""
            
            if provider == 'ollama':
                return await self._query_ollama_text(prompt_text, provider_settings)
            elif provider in ['openai', 'openrouter', 'anthropic']:
                return await self._query_api_text(prompt_text, provider, provider_settings)
            else:
                return await self._query_ollama_text(prompt_text, provider_settings)
        
        except Exception as e:
            logger.error(f"Text-only query error: {str(e)}")
            return f"Error processing query: {str(e)}"
    
    async def _query_ollama_text(self, prompt: str, settings_dict: Dict) -> str:
        """Query Ollama for text-only"""
        try:
            ollama_url = settings_dict.get('url', self.ollama_url)
            model = settings_dict.get('model', self.model_name)
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', 'No response from LLM')
                else:
                    return f"Error querying Ollama: {response.status_code}"
        
        except Exception as e:
            logger.error(f"Ollama text query error: {str(e)}")
            return f"Error querying Ollama: {str(e)}"
    
    async def _query_api_text(self, prompt: str, provider: str, settings_dict: Dict) -> str:
        """Query API providers (OpenAI, OpenRouter, Anthropic) for text-only"""
        try:
            api_key = settings_dict.get('apiKey', '')
            model = settings_dict.get('model', '')
            
            if not api_key:
                return f"Error: {provider} API key not configured"
            
            if provider == 'openai':
                url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": model or "gpt-4o",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000
                }
            elif provider == 'openrouter':
                url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/yourusername/document-qa",
                    "X-Title": "Document Q&A"
                }
                payload = {
                    "model": model or "google/gemini-pro-vision",
                    "messages": [{"role": "user", "content": prompt}]
                }
            elif provider == 'anthropic':
                url = "https://api.anthropic.com/v1/messages"
                headers = {
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": model or "claude-3-opus-20240229",
                    "max_tokens": 2000,
                    "messages": [{"role": "user", "content": prompt}]
                }
            else:
                return f"Unknown provider: {provider}"
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    if provider == 'anthropic':
                        return result['content'][0]['text']
                    else:
                        return result['choices'][0]['message']['content']
                else:
                    logger.error(f"{provider} error: {response.status_code} - {response.text}")
                    return f"Error querying {provider}: {response.status_code}"
        
        except Exception as e:
            logger.error(f"{provider} text query error: {str(e)}")
            return f"Error querying {provider}: {str(e)}"
    
    async def _query_with_llm_old(self, query: str, images_data: List[Dict]) -> str:
        """Query LLM with image data"""
        try:
            # Build messages for Ollama vision model
            messages = [{
                "role": "user",
                "content": query,
                "images": [img['base64'] for img in images_data if img.get('base64')]
            }]
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": self.model_name,
                        "messages": messages,
                        "stream": False
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('message', {}).get('content', 'No response from LLM')
                else:
                    logger.error(f"LLM error: {response.status_code} - {response.text}")
                    return f"Error querying LLM: {response.status_code}"
        
        except Exception as e:
            logger.error(f"LLM query error: {str(e)}")
            return f"Error querying LLM: {str(e)}"
    
    async def _query_text_only_old(self, query: str, page_docs: List) -> str:
        """Query with text-only (no images)"""
        try:
            # Extract text from documents
            context_parts = []
            for doc in page_docs:
                doc_text = getattr(doc, 'text', '') or getattr(doc, 'content', '')
                if doc_text:
                    context_parts.append(doc_text)
            
            context = "\n\n".join(context_parts[:3])  # Limit to 3 documents
            
            # Build prompt
            prompt = f"""Based on the following document context, please answer the question.

Context:
{context}

Question: {query}

Answer:"""
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', 'No response from LLM')
                else:
                    return f"Error querying LLM: {response.status_code}"
        
        except Exception as e:
            logger.error(f"Text-only query error: {str(e)}")
            return f"Error processing query: {str(e)}"
    
    async def general_knowledge_fallback(self, query: str, debug_info: List[str]) -> Dict:
        """Fallback to general knowledge when no documents found"""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": f"Please answer this question: {query}",
                        "stream": False
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get('response', 'Unable to generate response')
                else:
                    answer = "Error accessing general knowledge"
            
            debug_info.append("Used general knowledge fallback")
            
            return {
                "result": answer,
                "source_info": {"note": "Response based on general knowledge, not documents"},
                "debug_info": debug_info,
                "images_info": []
            }
        
        except Exception as e:
            logger.error(f"General fallback error: {str(e)}")
            return {
                "result": f"Unable to process query: {str(e)}",
                "source_info": {},
                "debug_info": debug_info,
                "images_info": []
            }
