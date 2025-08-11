# Gemini Deep Research Agent

A custom implementation of Google's Deep Research functionality using Gemini API, Google Search Grounding, and Crawl4AI for comprehensive research automation.

## 🔍 Project Overview

Since Google's Deep Research feature is **UI-only** (no API access), this project creates a custom agent that replicates and extends Deep Research capabilities programmatically.

### ❌ Google Deep Research API Status
- **No Direct API Available** - Exclusive to Gemini Advanced ($20/month)
- Only accessible via web/mobile interfaces
- Not available for programmatic integration

### ✅ Our Solution: Custom Deep Research Agent

**Core Components:**
1. **Google Search Grounding** - Gemini API integration ($35/1000 queries)
2. **Crawl4AI** - Open-source web scraping (free)
3. **Gemini 2.5 Pro/Flash** - Advanced reasoning models

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd gemini_deep_research

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers (required for Crawl4AI)
playwright install
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env
```

**Required Configuration:**
```env
# Your Gemini API key from https://ai.google.dev/
GEMINI_API_KEY=your_gemini_api_key_here

# Default model (recommended: gemini-2.5-pro)
GEMINI_MODEL=gemini-2.5-pro

# Search settings (10 searches = ~$0.35 per task)
MAX_SEARCHES_PER_TASK=10

# Crawling settings
MAX_CONCURRENT_CRAWLS=10
MAX_CRAWL_TIMEOUT_MINUTES=5
```

### 3. Usage

```python
from src.deep_research_agent import DeepResearchAgent, TokenCounter, ResearchConfig

# Initialize configuration from environment
config = ResearchConfig.from_env()

# Test the tokenizer
tokenizer = TokenCounter(config)
tokens = tokenizer.count_tokens("Your research query here")
print(f"Query tokens: {tokens}")

# Initialize the research agent (coming soon)
# agent = DeepResearchAgent(config)
# report = await agent.research("Latest developments in AI safety regulations 2024")
# print(report)
```

## 💰 Cost Analysis

### Pricing Structure
- **Gemini 2.5 Flash**: Most cost-effective, thinking capabilities
- **Gemini 2.5 Pro**: Advanced reasoning, best performance  
- **Google Search Grounding**: $35 per 1,000 search queries
- **Crawl4AI**: Free and open-source

### Default Cost per Task
- **10 searches** = $0.35 per task
- **Gemini API**: Free tier available, then pay-per-use
- **Total estimated cost**: $0.35 - $1.00 per research task

## 🏗️ Architecture

```python
class DeepResearchAgent:
    def __init__(self):
        self.gemini_client = genai.Client()      # Gemini API
        self.crawler = AsyncWebCrawler()         # Crawl4AI
        self.search_tool = GoogleSearch()        # Search grounding
    
    async def research(self, query: str) -> str:
        # 1. Intent extraction & keyword generation
        keywords = await self.extract_keywords(query)
        
        # 2. Multi-query search via Gemini grounding
        search_results = await self.search_multiple(keywords)
        
        # 3. URL extraction & content crawling
        urls = self.extract_urls(search_results)
        content = await self.crawl_urls(urls)
        
        # 4. Report synthesis with Gemini Pro 2.0
        report = await self.synthesize_report(query, content)
        
        return report
```

## ⚡ Performance Features

- **Async/Await Processing** - Maximum efficiency
- **Parallel URL Crawling** - Process multiple sources simultaneously
- **Smart Token Management** - Respects Gemini's 1M token context limit
- **Intelligent Timeouts** - 5-minute max with fallback to collected data
- **Thread Pool Optimization** - Configurable concurrent operations

## 🔧 Configuration Options

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_SEARCHES_PER_TASK` | 10 | Search queries per research task |
| `MAX_CONCURRENT_CRAWLS` | 10 | Parallel URL crawling limit |
| `GEMINI_MODEL` | gemini-2.5-pro | Primary model for research |
| `MAX_CRAWL_TIMEOUT_MINUTES` | 5 | Maximum crawling time |
| `GEMINI_CONTEXT_LIMIT` | 900000 | Token limit for context |

### Use Case Presets

**Budget-Conscious:**
```env
MAX_SEARCHES_PER_TASK=5
MAX_CONCURRENT_CRAWLS=5
GEMINI_MODEL=gemini-2.5-flash
```

**Comprehensive Research:**
```env
MAX_SEARCHES_PER_TASK=20
MAX_CONCURRENT_CRAWLS=15
GEMINI_MODEL=gemini-2.5-pro
```

**Development/Testing:**
```env
MAX_SEARCHES_PER_TASK=3
DEBUG_MODE=true
SAVE_INTERMEDIATE_RESULTS=true
```

## 🎯 Key Advantages

- ✅ **Cost-effective** - Free tier + transparent pricing
- ✅ **Full programmatic control** - API-based automation
- ✅ **Customizable research depth** - Configurable parameters
- ✅ **Multi-format output** - Markdown, HTML, JSON, TXT
- ✅ **Production-ready** - Built with enterprise tools
- ✅ **Open source** - Fully customizable and extensible

## 📋 Development Status

### ✅ Completed
- [x] Research Google Gemini Deep Research API availability
- [x] Analyze Google Search API integration with Gemini  
- [x] Evaluate Crawl4AI capabilities and integration
- [x] Design agent architecture for custom Deep Research system
- [x] Research Gemini Flash and Pro 2.0 API capabilities
- [x] Create comprehensive environment configuration
- [x] Document project setup and configuration

### ✅ Recently Completed
- [x] Implement advanced TokenCounter with multiple counting methods
- [x] Create comprehensive token budget management system
- [x] Add Gemini context window configuration variables
- [x] Build token usage monitoring and optimization
- [x] Create modular project structure with requirements.txt

### 🚧 In Progress  
- [ ] Implement core DeepResearchAgent class
- [ ] Create async search and crawling modules
- [ ] Add intelligent keyword extraction
- [ ] Build report synthesis engine
- [ ] Implement cost tracking and limits

### 📅 Planned
- [ ] Add result ranking and relevance scoring
- [ ] Implement caching system
- [ ] Create CLI interface
- [ ] Add batch processing capabilities
- [ ] Build web dashboard
- [ ] Add export formats (PDF, DOCX)

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines and submit pull requests for any improvements.

## 📄 License

This project is open source. See LICENSE file for details.

---

**Note**: This is an independent implementation and is not affiliated with Google or the official Gemini Deep Research feature.