"""
Main application routes blueprint.
Handles the main pages and system information endpoints.
"""
from flask import Blueprint, render_template, request, jsonify, session
import logging
import asyncio
from middleware.auth import login_required
from utils.network import get_local_ip, get_default_backend_url

logger = logging.getLogger(__name__)

# Create blueprint
main_bp = Blueprint('main', __name__)


@main_bp.route('/')
@login_required
def index():
    """Main application page"""
    return render_template('index.html')


@main_bp.route('/documents')
@login_required
def documents():
    """Documents management page"""
    return render_template('documents.html')


@main_bp.route('/api/system_info', methods=['GET'])
@login_required
def get_system_info():
    """Get system information including IP address"""
    try:
        local_ip = get_local_ip()
        default_backend = get_default_backend_url(local_ip)
        
        return jsonify({
            'success': True,
            'local_ip': local_ip,
            'suggested_url': f"http://{local_ip}:5001/query",
            'default_backend_url': default_backend,
            'user': {
                'username': session.get('username'),
                'role': session.get('role'),
                'email': session.get('email')
            }
        })
    except Exception as e:
        logger.error(f"System info error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to get system info'
        }), 500


@main_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'colivara-flask-app'
    }), 200


@main_bp.route('/api/status', methods=['GET'])
def api_status():
    """API status endpoint"""
    return jsonify({
        'success': True,
        'status': 'running',
        'version': '2.0.0',
        'features': [
            'authentication',
            'query_management',
            'document_management',
            'user_collections'
        ]
    })


@main_bp.route('/api/query', methods=['POST'])
@login_required
def query_documents():
    """Query documents endpoint with multi-provider support"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'detail': 'Query text is required'
            }), 400
        
        # Import here to avoid circular imports
        from api.services.query_service import QueryService
        from api.core.config import settings as api_settings
        
        # Get user's API key from session or use default
        api_key = session.get('api_key', api_settings.API_KEY)
        
        # Get provider settings from request
        provider = data.get('provider', 'ollama')
        provider_settings = data.get('provider_settings', {})
        
        # Initialize query service
        query_service = QueryService()
        
        # Process the query (run async function synchronously)
        result = asyncio.run(query_service.process_query(
            query=data['query'],
            api_key=api_key,
            collection_name=data.get('collection_name', 'my_collection'),
            include_next_page=data.get('include_next_page', False),
            max_additional_pages=data.get('max_additional_pages', 1),
            allow_general_fallback=data.get('allow_general_fallback', True),
            similarity_threshold=data.get('similarity_threshold', 0.25),
            provider=provider,
            provider_settings=provider_settings
        ))
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Query error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'detail': f'Query processing failed: {str(e)}'
        }), 500


@main_bp.route('/api/query-llm', methods=['POST'])
@login_required
def query_llm_only():
    """Pure LLM query endpoint without RAG/document retrieval"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'detail': 'Query text is required'
            }), 400
        
        import requests
        
        # Get provider settings from request
        provider = data.get('provider', 'ollama')
        provider_settings = data.get('provider_settings', {})
        query_text = data['query']
        
        # Call LLM directly without document context
        if provider == 'ollama':
            ollama_url = provider_settings.get('url', 'http://localhost:11434')
            model = provider_settings.get('model', 'qwen2.5vl:32b')
            
            response = requests.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": query_text}
                    ],
                    "stream": False
                },
                timeout=120
            )
            
            if response.ok:
                result = response.json()
                return jsonify({
                    'result': result.get('message', {}).get('content', 'No response'),
                    'mode': 'pure_llm',
                    'provider': provider
                })
            else:
                return jsonify({
                    'success': False,
                    'detail': f'LLM request failed: {response.text}'
                }), response.status_code
                
        elif provider == 'openai':
            import openai
            openai.api_key = provider_settings.get('apiKey', '')
            model = provider_settings.get('model', 'gpt-4o')
            
            completion = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": query_text}]
            )
            
            return jsonify({
                'result': completion.choices[0].message.content,
                'mode': 'pure_llm',
                'provider': provider
            })
            
        elif provider == 'anthropic':
            import anthropic
            client = anthropic.Anthropic(api_key=provider_settings.get('apiKey', ''))
            model = provider_settings.get('model', 'claude-3-opus-20240229')
            
            message = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": query_text}]
            )
            
            return jsonify({
                'result': message.content[0].text,
                'mode': 'pure_llm',
                'provider': provider
            })
            
        elif provider == 'openrouter':
            api_key = provider_settings.get('apiKey', '')
            model = provider_settings.get('model', 'anthropic/claude-3.5-sonnet')
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": query_text}]
                },
                timeout=120
            )
            
            if response.ok:
                result = response.json()
                return jsonify({
                    'result': result['choices'][0]['message']['content'],
                    'mode': 'pure_llm',
                    'provider': provider
                })
            else:
                return jsonify({
                    'success': False,
                    'detail': f'OpenRouter request failed: {response.text}'
                }), response.status_code
        else:
            return jsonify({
                'success': False,
                'detail': f'Unsupported provider: {provider}'
            }), 400
            
    except Exception as e:
        logger.error(f"LLM query error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'detail': f'LLM query failed: {str(e)}'
        }), 500


@main_bp.route('/api/documents/upload', methods=['POST'])
@login_required
def upload_document():
    """Upload document to ColiVara API (port 5001)"""
    try:
        import requests
        from werkzeug.utils import secure_filename
        
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'detail': 'No file provided'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'detail': 'No file selected'
            }), 400
        
        collection_name = request.form.get('collection_name', 'default_collection')
        
        # Get local IP for API URL
        local_ip = get_local_ip()
        api_url = f"http://{local_ip}:5001/api/upload"
        
        # Prepare the file for upload
        files = {'file': (file.filename, file.stream, file.content_type)}
        data = {'collection_name': collection_name}
        
        # Forward to FastAPI on port 5001
        response = requests.post(api_url, files=files, data=data, timeout=300)
        
        # Check if response has content
        if not response.content:
            logger.error(f"Empty response from API at {api_url}, status: {response.status_code}")
            return jsonify({
                'success': False,
                'detail': f'API service on port 5001 returned empty response (status {response.status_code}). Please check if the FastAPI service is running properly.'
            }), 503
        
        if response.status_code in [200, 201]:
            try:
                return jsonify(response.json()), 200
            except:
                return jsonify({
                    'success': False,
                    'detail': f'API returned invalid JSON response: {response.text[:200]}'
                }), 500
        else:
            # Try to parse JSON error, fallback to text if it fails
            try:
                error_detail = response.json().get('detail', 'Upload failed')
            except:
                error_detail = f'Upload failed (HTTP {response.status_code}): {response.text[:200] or "No error message"}'
            
            return jsonify({
                'success': False,
                'detail': error_detail
            }), response.status_code
            
    except requests.exceptions.Timeout:
        logger.error("Upload timeout")
        return jsonify({
            'success': False,
            'detail': 'Upload timeout - file may be too large'
        }), 504
    except requests.exceptions.ConnectionError:
        logger.error(f"Cannot connect to API at {api_url}")
        return jsonify({
            'success': False,
            'detail': 'Cannot connect to API service on port 5001. Please ensure the FastAPI service is running.'
        }), 503
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'detail': f'Upload failed: {str(e)}'
        }), 500


@main_bp.route('/api/documents/list', methods=['GET'])
@login_required
def list_documents():
    """List documents from ColiVara API (port 5001)"""
    try:
        import requests
        
        collection_name = request.args.get('collection_name', 'default_collection')
        
        # Get local IP for API URL
        local_ip = get_local_ip()
        api_url = f"http://{local_ip}:5001/api/documents"
        
        # Forward to FastAPI on port 5001
        response = requests.get(
            api_url,
            params={'collection_name': collection_name},
            timeout=30
        )
        
        # Check if response has content
        if not response.content:
            logger.error(f"Empty response from API at {api_url}, status: {response.status_code}")
            return jsonify({
                'success': False,
                'detail': f'API service on port 5001 returned empty response (status {response.status_code}). Please check if the FastAPI service is running properly.',
                'documents': []
            }), 503
        
        if response.status_code == 200:
            try:
                return jsonify(response.json()), 200
            except:
                return jsonify({
                    'success': False,
                    'detail': f'API returned invalid JSON: {response.text[:200]}',
                    'documents': []
                }), 500
        else:
            # Try to parse JSON error, fallback to text if it fails
            try:
                error_detail = response.json().get('detail', 'Failed to list documents')
            except:
                error_detail = f'Failed to list documents (HTTP {response.status_code}): {response.text[:200] or "No error message"}'
            
            return jsonify({
                'success': False,
                'detail': error_detail
            }), response.status_code
            
    except requests.exceptions.Timeout:
        logger.error("List documents timeout")
        return jsonify({
            'success': False,
            'detail': 'Request timeout - API service took too long to respond',
            'documents': []
        }), 504
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Cannot connect to API at {api_url}: {str(e)}")
        return jsonify({
            'success': False,
            'detail': 'Cannot connect to API service on port 5001. Please ensure the FastAPI service is running.',
            'documents': []
        }), 503
    except Exception as e:
        logger.error(f"List documents error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'detail': f'Failed to list documents: {str(e)}'
        }), 500


@main_bp.route('/api/documents/delete/<path:filename>', methods=['DELETE'])
@login_required
def delete_document(filename):
    """Delete document from ColiVara API (port 5001)"""
    try:
        import requests
        import urllib.parse
        
        collection_name = request.args.get('collection_name', 'default_collection')
        
        # Get local IP for API URL
        local_ip = get_local_ip()
        
        # URL encode the filename
        encoded_filename = urllib.parse.quote(filename, safe='')
        api_url = f"http://{local_ip}:5001/api/delete/{encoded_filename}"
        
        # Forward to FastAPI on port 5001
        response = requests.delete(
            api_url,
            params={'collection_name': collection_name},
            timeout=30
        )
        
        if response.status_code in [200, 204]:
            return jsonify({
                'success': True,
                'message': f'Document "{filename}" deleted successfully',
                'filename': filename
            }), 200
        elif response.status_code == 404:
            return jsonify({
                'success': False,
                'detail': f'Document "{filename}" not found'
            }), 404
        else:
            return jsonify({
                'success': False,
                'detail': response.json().get('detail', 'Delete failed')
            }), response.status_code
            
    except requests.exceptions.Timeout:
        logger.error("Delete document timeout")
        return jsonify({
            'success': False,
            'detail': 'Request timeout'
        }), 504
    except Exception as e:
        logger.error(f"Delete document error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'detail': f'Failed to delete document: {str(e)}'
        }), 500

