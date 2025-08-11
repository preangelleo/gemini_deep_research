#!/usr/bin/env python3
"""
Gemini Deep Research Agent - Flask API Server
============================================

A REST API wrapper for the Gemini Deep Research Agent, providing
easy access to comprehensive AI-powered research capabilities.

Features:
- 🔍 Intelligent search strategy generation
- 🌐 Google Search with Gemini Grounding  
- 🕸️ Clean web content extraction
- 🤖 AI-powered content polishing
- 📄 Multiple report formats (JSON, Markdown)

Author: Deep Research Team
Version: 1.0.0
Port: 5357 (chosen to avoid common port conflicts)
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import traceback
from functools import wraps

# Flask imports
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import threading
from concurrent.futures import ThreadPoolExecutor
import time

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'prompts'))

# Import our research agent
try:
    from deep_research_agent import (
        ResearchConfig, TokenCounter, KeywordExtractor,
        GeminiSearchEngine, WebContentExtractor, AIContentPolisher
    )
    from research_analysis_reporter import get_system_prompt, get_available_prompt_types
except ImportError as e:
    print(f"❌ Failed to import research components: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app configuration
app = Flask(__name__)
CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000'])  # Allow frontend access
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# Configuration
API_PORT = int(os.getenv('API_PORT', '5357'))
API_HOST = os.getenv('API_HOST', '0.0.0.0')
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

# Thread executor for async operations
executor = ThreadPoolExecutor(max_workers=4)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def async_to_sync(async_func):
    """Convert async function to sync for Flask compatibility."""
    @wraps(async_func)
    def wrapper(*args, **kwargs):
        try:
            # Create new event loop for this thread if needed
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    raise RuntimeError("Event loop is closed")
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(async_func(*args, **kwargs))
        except Exception as e:
            logger.error(f"Async execution error: {e}")
            raise
    return wrapper

def validate_api_key():
    """Validate that Gemini API key is configured."""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'your_gemini_api_key_here':
        return False, "GEMINI_API_KEY not configured. Please set it in your environment variables."
    return True, ""

def create_error_response(message: str, status_code: int = 400, error_type: str = "error") -> tuple:
    """Create standardized error response."""
    return jsonify({
        "success": False,
        "error": {
            "type": error_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "status_code": status_code
        }
    }), status_code

def create_success_response(data: Dict[str, Any], message: str = "Success") -> Dict[str, Any]:
    """Create standardized success response."""
    return {
        "success": True,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "data": data
    }

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/', methods=['GET'])
def home():
    """API home endpoint with usage information."""
    return jsonify({
        "service": "Gemini Deep Research Agent API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "/health": "Health check",
            "/config": "Configuration status",
            "/research": "Main research endpoint (POST)",
            "/research/stream": "Streaming research endpoint (POST)",
            "/prompts": "Available prompt styles",
            "/examples": "Usage examples"
        },
        "documentation": {
            "github": "https://github.com/preangelleo/gemini_deep_research",
            "docker": "docker run -p 5357:5357 betashow/gemini-deep-research",
            "port": API_PORT
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    is_valid, error_msg = validate_api_key()
    
    # Test component initialization
    try:
        config = ResearchConfig.from_env()
        components_status = {
            "gemini_api": bool(config.gemini_api_key),
            "token_counter": True,
            "keyword_extractor": True,
            "search_engine": bool(config.gemini_api_key),
            "content_extractor": True,
            "ai_polisher": bool(config.gemini_api_key)
        }
    except Exception as e:
        components_status = {"error": str(e)}
    
    return jsonify({
        "status": "healthy" if is_valid else "configuration_needed",
        "timestamp": datetime.now().isoformat(),
        "components": components_status,
        "configuration": {
            "api_key_configured": is_valid,
            "error": error_msg if not is_valid else None
        }
    })

@app.route('/config', methods=['GET'])
def get_config():
    """Get current configuration status."""
    try:
        config = ResearchConfig.from_env()
        return jsonify(create_success_response({
            "gemini_model": config.gemini_model,
            "ai_polish_model": config.ai_polish_model,
            "max_searches_per_task": config.max_searches_per_task,
            "max_concurrent_crawls": config.max_concurrent_crawls,
            "ai_polish_content": config.ai_polish_content,
            "api_configured": bool(config.gemini_api_key)
        }))
    except Exception as e:
        return create_error_response(f"Configuration error: {str(e)}", 500, "config_error")

@app.route('/prompts', methods=['GET'])
def get_available_prompts():
    """Get available system prompt styles."""
    try:
        prompt_types = get_available_prompt_types()
        return jsonify(create_success_response({
            "available_styles": prompt_types,
            "default": "comprehensive",
            "descriptions": {
                "comprehensive": "Detailed research reports with analysis",
                "executive": "Concise summaries for decision-makers",
                "academic": "Literature reviews with citations",
                "technical": "Implementation-focused with specifications",
                "market": "Competitive intelligence and business analysis"
            }
        }))
    except Exception as e:
        return create_error_response(f"Prompts error: {str(e)}", 500)

@app.route('/examples', methods=['GET'])
def get_examples():
    """Get usage examples."""
    return jsonify(create_success_response({
        "basic_research": {
            "method": "POST",
            "url": "/research",
            "body": {
                "query": "Latest AI safety regulations 2024",
                "max_searches": 5,
                "ai_polish": True
            }
        },
        "comprehensive_report": {
            "method": "POST", 
            "url": "/research",
            "body": {
                "query": "Quantum computing breakthroughs",
                "max_searches": 10,
                "ai_polish": 2,
                "output_format": "markdown"
            }
        },
        "streaming_research": {
            "method": "POST",
            "url": "/research/stream", 
            "body": {
                "query": "Climate change solutions 2024",
                "max_searches": 8
            }
        }
    }))

@app.route('/research', methods=['POST'])
def research_endpoint():
    """Main research endpoint."""
    try:
        # Validate API key
        is_valid, error_msg = validate_api_key()
        if not is_valid:
            return create_error_response(error_msg, 401, "authentication_error")
        
        # Parse request
        data = request.get_json()
        if not data:
            return create_error_response("Request body must be JSON", 400, "invalid_request")
        
        # Extract parameters
        query = data.get('query', '').strip()
        if not query:
            return create_error_response("Query parameter is required", 400, "missing_parameter")
        
        max_searches = data.get('max_searches', 5)
        max_crawls = data.get('max_crawls', 3)
        ai_polish = data.get('ai_polish', False)
        output_format = data.get('output_format', 'json').lower()
        prompt_style = data.get('prompt_style', 'comprehensive')
        
        # Validate parameters
        if not isinstance(max_searches, int) or max_searches < 1 or max_searches > 20:
            return create_error_response("max_searches must be integer between 1-20", 400, "invalid_parameter")
        
        if not isinstance(max_crawls, int) or max_crawls < 1 or max_crawls > 10:
            return create_error_response("max_crawls must be integer between 1-10", 400, "invalid_parameter")
        
        if output_format not in ['json', 'markdown']:
            return create_error_response("output_format must be 'json' or 'markdown'", 400, "invalid_parameter")
        
        # Execute research
        logger.info(f"Starting research: '{query}' (searches={max_searches}, crawls={max_crawls}, polish={ai_polish})")
        
        result = executor.submit(_execute_research, {
            'query': query,
            'max_searches': max_searches,
            'max_crawls': max_crawls,
            'ai_polish': ai_polish,
            'output_format': output_format,
            'prompt_style': prompt_style
        }).result(timeout=600)  # 10 minute timeout
        
        if output_format == 'markdown' and result.get('polished_content'):
            return Response(
                result['polished_content'],
                mimetype='text/markdown',
                headers={
                    'Content-Disposition': f'attachment; filename=research_{int(time.time())}.md',
                    'X-Research-Query': query,
                    'X-Processing-Time': str(result.get('processing_time', 0))
                }
            )
        
        return jsonify(create_success_response(result, "Research completed successfully"))
        
    except TimeoutError:
        return create_error_response("Research timeout - try reducing search scope", 504, "timeout_error")
    except Exception as e:
        logger.error(f"Research endpoint error: {str(e)}\n{traceback.format_exc()}")
        return create_error_response(f"Research failed: {str(e)}", 500, "research_error")

@async_to_sync
async def _execute_research(params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute research with given parameters."""
    start_time = datetime.now()
    
    try:
        # Initialize configuration
        config = ResearchConfig.from_env()
        config.max_searches_per_task = params['max_searches']
        config.max_concurrent_crawls = params['max_crawls']
        config.ai_polish_content = params['ai_polish']
        config.debug_mode = DEBUG_MODE
        
        # Initialize components
        tokenizer = TokenCounter(config)
        keyword_extractor = KeywordExtractor(config)
        search_engine = GeminiSearchEngine(config, tokenizer)
        content_extractor = WebContentExtractor(config, tokenizer)
        ai_polisher = AIContentPolisher(config, tokenizer) if params['ai_polish'] else None
        
        # Step 1: Generate search strategy
        strategy = keyword_extractor.generate_search_strategy(params['query'])
        logger.info(f"Generated search strategy with complexity {strategy['complexity_score']:.2f}")
        
        # Step 2: Execute searches
        search_queries = strategy['search_phases']['primary'][:params['max_searches']]
        search_results = await search_engine.search_multiple_queries(search_queries)
        logger.info(f"Found {len(search_results)} search results")
        
        # Step 3: Extract content
        top_results = search_results[:params['max_crawls'] * 2]
        content_results = await content_extractor.extract_content_with_strategy(top_results)
        crawled_content = content_results['crawled_content']
        successful_content = [c for c in crawled_content if c.success]
        logger.info(f"Successfully extracted content from {len(successful_content)} sources")
        
        # Step 4: AI Content Polishing (if enabled)
        polished_content = None
        if params['ai_polish'] and ai_polisher and successful_content:
            logger.info("Starting AI content polishing...")
            polished_content = await ai_polisher.polish_content(
                params['query'], search_results, crawled_content
            )
            logger.info(f"AI polishing completed: {len(polished_content) if polished_content else 0} characters")
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Prepare response
        result = {
            'query': params['query'],
            'processing_time': processing_time,
            'search_strategy': {
                'complexity_score': strategy['complexity_score'],
                'primary_queries': strategy['search_phases']['primary'][:5],  # Limit for response size
                'intent_analysis': strategy['intent_analysis']
            },
            'search_results': {
                'total_found': len(search_results),
                'sources_crawled': len(top_results),
                'successful_extractions': len(successful_content),
                'results': [
                    {
                        'title': r.title,
                        'url': r.url,
                        'relevance_score': r.relevance_score,
                        'source_type': r.source_type
                    } for r in search_results[:10]  # Limit for response size
                ]
            },
            'content_analysis': {
                'total_content_length': content_results['total_content_length'],
                'total_tokens': content_results['total_token_count'],
                'sources': [
                    {
                        'url': c.url,
                        'title': c.title,
                        'length': c.content_length,
                        'tokens': c.token_count,
                        'success': c.success,
                        'crawl_time': c.crawl_time
                    } for c in crawled_content[:10]  # Limit for response size
                ]
            },
            'ai_polishing': {
                'enabled': bool(params['ai_polish']),
                'model_used': config.ai_polish_model if params['ai_polish'] else None,
                'polished_length': len(polished_content) if polished_content else 0
            },
            'polished_content': polished_content,
            'raw_content': '\n\n'.join([c.content for c in successful_content[:3]]) if not polished_content else None,  # Fallback
            'metadata': {
                'timestamp': start_time.isoformat(),
                'configuration': {
                    'max_searches': params['max_searches'],
                    'max_crawls': params['max_crawls'],
                    'ai_polish_level': params['ai_polish'],
                    'gemini_model': config.gemini_model
                }
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Research execution failed: {str(e)}\n{traceback.format_exc()}")
        raise

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found(error):
    return create_error_response("Endpoint not found", 404, "not_found")

@app.errorhandler(405)
def method_not_allowed(error):
    return create_error_response("Method not allowed", 405, "method_not_allowed")

@app.errorhandler(500)
def internal_error(error):
    return create_error_response("Internal server error", 500, "internal_error")

# =============================================================================
# STARTUP
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🔬 GEMINI DEEP RESEARCH AGENT - API SERVER")
    print("="*70)
    print(f"🌐 Starting server on {API_HOST}:{API_PORT}")
    print(f"🤖 Debug mode: {DEBUG_MODE}")
    print(f"📚 Documentation: http://localhost:{API_PORT}/")
    print(f"🏥 Health check: http://localhost:{API_PORT}/health")
    print(f"🧪 Examples: http://localhost:{API_PORT}/examples")
    print("="*70)
    
    # Validate configuration on startup
    is_valid, error_msg = validate_api_key()
    if not is_valid:
        print(f"⚠️  WARNING: {error_msg}")
        print(f"   The API will return authentication errors until this is fixed.")
    else:
        print("✅ Gemini API key configured successfully")
    
    print("\n🚀 Server starting...\n")
    
    try:
        app.run(
            host=API_HOST,
            port=API_PORT,
            debug=DEBUG_MODE,
            threaded=True,
            use_reloader=False  # Disable auto-reload to prevent duplicate startup messages
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Server shutdown requested by user")
    except Exception as e:
        print(f"\n❌ Server startup failed: {e}")