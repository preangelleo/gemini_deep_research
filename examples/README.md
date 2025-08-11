# Gemini Deep Research Agent - Examples & Demos

This directory contains usage examples and interactive demonstrations for the Gemini Deep Research Agent.

## 📋 Quick Start

### Main Usage Example
```bash
python examples/usage_example.py
```
**Complete research workflow with customizable parameters. Perfect for production use.**

## 🧪 Interactive Demos

### 1. Search Engine Demo
```bash
python examples/interactive_search_demo.py
```
**Test Google Search with Gemini Grounding:**
- Input any search query interactively
- See detailed search results with relevance scores  
- Understand token usage and costs (~$0.035 per search)
- Results automatically saved for analysis

### 2. Web Crawler Demo  
```bash
python examples/interactive_crawler_demo.py
```
**Test clean content extraction from any URL:**
- Input any website URL to test extraction quality
- Get clean markdown without messy hyperlinks
- Paywall detection and content quality analysis
- Support for international content (Chinese, etc.)

### 3. AI Polishing Demo
```bash
python examples/ai_polishing_demo.py
```
**Complete workflow with AI enhancement:**
- Full research pipeline (search → extract → polish)
- AI-powered content polishing with Gemini 2.5 Pro
- Blog-ready comprehensive report generation
- Performance metrics and cost tracking

## 📁 File Descriptions

| File | Purpose | Use Case |
|------|---------|----------|
| `usage_example.py` | Main production example | Copy and customize for your projects |
| `interactive_search_demo.py` | Search testing | Test search queries and analyze results |  
| `interactive_crawler_demo.py` | URL extraction testing | Test content extraction from specific URLs |
| `ai_polishing_demo.py` | Complete workflow demo | See full AI research pipeline in action |

## 🔧 Configuration

All examples use your `.env` file configuration. Key settings:

```env
GEMINI_API_KEY=your_api_key_here
AI_POLISH_CONTENT=2                    # Enable comprehensive reports
MAX_SEARCHES_PER_TASK=5               # Balance cost vs depth
MAX_CONCURRENT_CRAWLS=3               # Parallel processing
```

## 🎨 Customizing Report Style

**NEW FEATURE**: Customize your AI-generated reports by editing the system prompts!

### How to Customize Report Style:
1. **Edit the system prompt**: `/prompts/research_analysis_reporter.py`
2. **Modify the `SYSTEM_PROMPT_RESEARCH_ANALYSIS` variable**
3. **Run your research** - the new style will be applied automatically

### Available Report Styles:
- **Comprehensive** (default) - Detailed research reports with analysis
- **Executive** - Concise summaries for decision-makers
- **Academic** - Literature reviews with citations and methodology  
- **Technical** - Implementation-focused with specifications
- **Market** - Competitive intelligence and business analysis

### Example Customization:
```python
# In prompts/research_analysis_reporter.py
# Change this line to use a different style:
DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPT_EXECUTIVE_SUMMARY  # For executive style
```

**💡 TIP**: The system prompt controls the structure, tone, and focus of your AI-generated reports. Experiment with different prompts to find your preferred style!

## 💡 Usage Tips

### For Production Use:
- Start with `usage_example.py` and customize parameters
- Set appropriate search limits to control costs
- Enable AI polishing for professional reports

### For Testing:
- Use individual demos to test specific components
- `interactive_search_demo.py` - Test search quality
- `interactive_crawler_demo.py` - Test URL extraction
- `ai_polishing_demo.py` - Test complete workflow

### Cost Optimization:
- Each search costs ~$0.035 (Google Grounding)
- Reduce `max_searches` for budget-conscious research
- Use `ai_polish_level=False` to disable AI enhancement

## 📊 Expected Results

### Search Demo:
- 5-10 high-quality search results per query
- Relevance scores and source type classification
- ~2 second execution time per search

### Crawler Demo:  
- Clean markdown extraction from most websites
- Paywall detection and content quality grading
- Support for news, articles, research papers, etc.

### AI Polishing Demo:
- 2-5 minute complete workflow execution
- Professional blog-ready research reports
- Comprehensive analysis with citations and structure

## 🚀 Next Steps

1. **Try the demos** to understand each component
2. **Customize usage_example.py** for your specific needs  
3. **Integrate into your projects** using the provided patterns
4. **Scale up** search intensity and AI polishing as needed

## 📝 Output Files

### Main Output (What You Actually Want):
- **`report.md`** - Clean research report (when AI polishing enabled)
- This is the main file users need - ready to read/publish

### Additional Files:
- `demo_results/` - Interactive demo outputs  
- `research_results/` - Production research outputs
- `*.json` - Complete technical data (for developers/debugging)
- `*_timestamp.md` - Archived copies with timestamps

**💡 TIP:** When using AI polishing, you mainly want the `report.md` file - it contains the clean, professional research report without technical metadata.

## ⚠️ Requirements

Ensure you have:
- ✅ GEMINI_API_KEY configured in .env
- ✅ Dependencies installed: `pip install -r requirements.txt`  
- ✅ Playwright browsers: `playwright install`

## 🆘 Troubleshooting

**"Gemini API not configured"**: Check your .env file has GEMINI_API_KEY  
**"Crawl4AI not initialized"**: Run `pip install crawl4ai && playwright install`  
**"Search results empty"**: Try different search queries or check API quotas  
**"Content extraction failed"**: Test with different URLs, some sites block crawlers  

---

🔬 **Ready to start researching?** Run `python examples/usage_example.py` and customize for your needs!