# Flask API - Gemini Deep Research Agent

This document explains how to run and use the Flask API server for the Gemini Deep Research Agent locally.

## 🚀 Quick Start

### Prerequisites

- Python 3.11+ (recommended for best compatibility)
- Gemini API key from [Google AI Studio](https://ai.google.dev/)

### Installation

```bash
# Clone the repository
git clone https://github.com/preangelleo/gemini_deep_research.git
cd gemini_deep_research

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install

# Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### Starting the Server

```bash
# Start the Flask API server
python app.py
```

The server will start on **http://localhost:5357**

## 📋 API Endpoints

### Health Check
```bash
GET /health
```
Check if the API is running and properly configured.

### Research Endpoint
```bash
POST /research
Content-Type: application/json

{
  "query": "Latest AI safety regulations 2024",
  "max_searches": 5,
  "max_crawls": 3,
  "ai_polish": 2,
  "output_format": "json"
}
```

### Configuration
```bash
GET /config
```
Get current configuration status.

### Available Prompts
```bash
GET /prompts
```
List available system prompt styles.

## 🔧 API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Your research question |
| `max_searches` | integer | 5 | Number of search queries (1-20) |
| `max_crawls` | integer | 3 | Number of URLs to crawl (1-10) |
| `ai_polish` | boolean/integer | false | AI polishing level (false, true, 2) |
| `output_format` | string | "json" | Response format ("json" or "markdown") |
| `prompt_style` | string | "comprehensive" | Report style |

### AI Polish Levels:
- `false` - No AI polishing
- `true` - Basic content cleaning
- `2` - Comprehensive report generation

### Available Prompt Styles:
- `comprehensive` - Detailed research reports (default)
- `executive` - Concise business summaries  
- `academic` - Literature reviews with citations
- `technical` - Implementation-focused reports
- `market` - Competitive intelligence analysis

## 💡 Usage Examples

### Basic Research
```bash
curl -X POST http://localhost:5357/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Quantum computing breakthroughs 2024",
    "max_searches": 5,
    "ai_polish": true
  }'
```

### Comprehensive Report Generation
```bash
curl -X POST http://localhost:5357/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Climate change solutions renewable energy",
    "max_searches": 10,
    "max_crawls": 5,
    "ai_polish": 2,
    "prompt_style": "comprehensive",
    "output_format": "markdown"
  }'
```

### Executive Summary
```bash
curl -X POST http://localhost:5357/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "AI impact on job market 2024",
    "max_searches": 8,
    "ai_polish": 2,
    "prompt_style": "executive"
  }'
```

## 📊 Response Format

### JSON Response
```json
{
  "success": true,
  "message": "Research completed successfully",
  "timestamp": "2024-08-11T15:30:45.123456",
  "data": {
    "query": "Your research query",
    "processing_time": 45.67,
    "search_strategy": {
      "complexity_score": 2.8,
      "primary_queries": ["query1", "query2", ...],
      "intent_analysis": {...}
    },
    "search_results": {
      "total_found": 15,
      "sources_crawled": 6,
      "successful_extractions": 5,
      "results": [...]
    },
    "content_analysis": {
      "total_content_length": 12450,
      "total_tokens": 3112,
      "sources": [...]
    },
    "ai_polishing": {
      "enabled": true,
      "model_used": "gemini-2.5-pro",
      "polished_length": 2847
    },
    "polished_content": "# Research Report...",
    "metadata": {...}
  }
}
```

### Markdown Response
When `output_format: "markdown"` is used, the API returns the polished content as a downloadable markdown file.

## ⚙️ Configuration

### Environment Variables
Set these in your `.env` file:

```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional API Configuration
API_HOST=0.0.0.0                    # Server host
API_PORT=5357                       # Server port
DEBUG_MODE=false                    # Enable debug logging

# Research Configuration  
MAX_SEARCHES_PER_TASK=10           # Default max searches
MAX_CONCURRENT_CRAWLS=5            # Default max crawls
AI_POLISH_CONTENT=false            # Default AI polishing
GEMINI_MODEL=gemini-2.5-pro       # Gemini model to use
```

### Custom System Prompts
Customize report styles by editing:
```
prompts/research_analysis_reporter.py
```

Change the `DEFAULT_SYSTEM_PROMPT` variable to use different styles.

## 🔍 Error Handling

### Common Error Responses
```json
{
  "success": false,
  "error": {
    "type": "authentication_error",
    "message": "GEMINI_API_KEY not configured",
    "timestamp": "2024-08-11T15:30:45.123456",
    "status_code": 401
  }
}
```

### Error Types:
- `authentication_error` - Missing or invalid API key
- `invalid_request` - Malformed request
- `invalid_parameter` - Parameter validation failed
- `research_error` - Research execution failed
- `timeout_error` - Request timed out (10 min limit)

## 🚀 Performance Tips

### For Better Performance:
1. **Limit search scope** - Use fewer `max_searches` for faster results
2. **Optimize crawling** - Reduce `max_crawls` if speed is priority
3. **Use appropriate AI polishing** - Level 2 takes longer but produces better results
4. **Choose right prompt style** - Executive style is faster than comprehensive

### Cost Optimization:
- Each search costs ~$0.035 (Google Search Grounding)
- AI polishing uses additional Gemini tokens
- Monitor usage with the `/config` endpoint

## 🧪 Development

### Running in Development Mode
```bash
export DEBUG_MODE=true
python app.py
```

### Testing API Endpoints
```bash
# Health check
curl http://localhost:5357/health

# Get configuration
curl http://localhost:5357/config

# List available prompts
curl http://localhost:5357/prompts

# Get usage examples
curl http://localhost:5357/examples
```

### Adding Custom Endpoints
Edit `app.py` and add new routes following the existing patterns.

## 🐛 Troubleshooting

### Common Issues:

**"GEMINI_API_KEY not configured"**
- Add your API key to `.env` file
- Get key from https://ai.google.dev/

**"Module not found" errors**
- Run `pip install -r requirements.txt`
- Check Python version (3.11+ recommended)

**"Playwright browser not found"**
- Run `playwright install`
- Check system dependencies

**Slow response times**
- Reduce `max_searches` and `max_crawls`
- Disable AI polishing for faster results
- Check network connection

**Port already in use**
- Change `API_PORT` in `.env` file
- Kill existing processes on port 5357

### Getting Help:
- Check the logs for detailed error messages
- Visit the [GitHub repository](https://github.com/preangelleo/gemini_deep_research)
- Use the `/health` endpoint to diagnose configuration issues

---

## 🔗 Related Documentation

- [Docker Usage Guide](DOCKER.md) - Run with Docker
- [Main README](../README.md) - Project overview
- [Examples](../examples/) - Usage examples
- [System Prompts](../prompts/) - Customization guide