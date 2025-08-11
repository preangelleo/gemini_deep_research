#!/usr/bin/env python3
"""
AI CONTENT POLISHING - COMPREHENSIVE DEMO
=========================================

Test the complete research workflow with AI-powered content polishing.
This demo shows the full pipeline:
- Intelligent search strategy generation
- Google Search with Gemini Grounding  
- Clean web content extraction
- AI content polishing with Gemini 2.5 Pro
- Blog-ready comprehensive report generation

WHAT TO EXPECT:
- Complete research workflow (2-5 minutes)
- Professional report generation
- Cost tracking and optimization
- Blog-ready markdown output
- Comprehensive analysis and metrics

USAGE:
python examples/ai_polishing_demo.py
"""

import sys
import os
import asyncio
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from deep_research_agent import (
    AIContentPolisher,
    KeywordExtractor,
    GeminiSearchEngine,
    WebContentExtractor,
    ResearchConfig,
    TokenCounter
)

def get_demo_preferences():
    """Get user preferences for the AI polishing demo."""
    print("🤖 AI CONTENT POLISHING - COMPREHENSIVE DEMO")
    print("=" * 70)
    print("Experience the complete research workflow with AI enhancement")
    print("=" * 70)
    
    # Research topics
    topics = [
        ("AI Safety", "AI safety regulations impact on tech companies 2024"),
        ("Climate Tech", "Latest climate change solutions and green technology 2024"),
        ("Quantum Computing", "Quantum computing breakthroughs and commercial applications"),
        ("Space Exploration", "Mars exploration missions and space technology advances"), 
        ("Renewable Energy", "Solar and wind energy efficiency improvements 2024")
    ]
    
    print(f"\n📋 RESEARCH TOPICS:")
    for i, (category, topic) in enumerate(topics, 1):
        print(f"  {i}. {category}: {topic}")
    
    # Get topic selection
    user_choice = input(f"\n📝 Choose topic (1-{len(topics)}) or enter custom query: ").strip()
    
    if user_choice.isdigit():
        num = int(user_choice)
        if 1 <= num <= len(topics):
            research_query = topics[num - 1][1]
            print(f"✓ Selected: {topics[num - 1][0]}")
        else:
            research_query = topics[0][1]
            print(f"✓ Invalid selection, using default: {topics[0][0]}")
    elif user_choice:
        research_query = user_choice
        print(f"✓ Custom query: {research_query}")
    else:
        research_query = topics[0][1]
        print(f"✓ Using default: {topics[0][0]}")
    
    # AI polishing options
    print(f"\n🤖 AI POLISHING LEVELS:")
    print(f"1. Basic polishing - Clean and organize content")
    print(f"2. Comprehensive report - Full blog-ready article")
    print(f"3. Disabled - Raw extracted content only")
    
    polish_choice = input("Choose AI polish level (1-3, default=2): ").strip()
    if polish_choice == "1":
        polish_level = True
        print("✓ Basic AI polishing enabled")
    elif polish_choice == "3":
        polish_level = False
        print("✓ AI polishing disabled")
    else:
        polish_level = 2
        print("✓ Comprehensive report generation enabled")
    
    # Research intensity
    print(f"\n🔍 RESEARCH INTENSITY:")
    print(f"1. Light research (2 searches, ~$0.07)")
    print(f"2. Standard research (5 searches, ~$0.18)")
    print(f"3. Deep research (10 searches, ~$0.35)")
    
    intensity_choice = input("Choose research intensity (1-3, default=2): ").strip()
    if intensity_choice == "1":
        max_searches = 2
        max_crawls = 2
    elif intensity_choice == "3":
        max_searches = 10
        max_crawls = 5
    else:
        max_searches = 5
        max_crawls = 3
    
    estimated_cost = max_searches * 0.035
    print(f"✓ {max_searches} searches, ~${estimated_cost:.3f} estimated cost")
    
    return research_query, polish_level, max_searches, max_crawls

async def run_ai_polishing_demo():
    """Run complete AI polishing demonstration."""
    
    # Get user preferences
    research_query, polish_level, max_searches, max_crawls = get_demo_preferences()
    
    print(f"\n🔧 INITIALIZING AI RESEARCH SYSTEM...")
    
    # Initialize configuration
    config = ResearchConfig.from_env()
    config.ai_polish_content = polish_level
    config.debug_mode = True
    config.max_searches_per_task = max_searches
    config.max_concurrent_crawls = max_crawls
    
    # Validate API key
    if not config.gemini_api_key:
        print("❌ ERROR: GEMINI_API_KEY not found in .env file")
        return False
    
    # Initialize components
    tokenizer = TokenCounter(config)
    keyword_extractor = KeywordExtractor(config)
    search_engine = GeminiSearchEngine(config, tokenizer)
    content_extractor = WebContentExtractor(config, tokenizer)
    ai_polisher = AIContentPolisher(config, tokenizer) if polish_level else None
    
    # Validate initialization
    if not search_engine.client:
        print("❌ Gemini API not configured for search")
        return False
    
    if polish_level and not ai_polisher.client:
        print("❌ Gemini API not configured for AI polishing")
        return False
    
    if not content_extractor.crawler_initialized:
        print("❌ Crawl4AI not initialized")
        print("Run: pip install crawl4ai && playwright install")
        return False
    
    print("✅ All components initialized successfully!")
    print(f"🤖 AI Polish: {'Comprehensive' if polish_level == 2 else 'Basic' if polish_level else 'Disabled'}")
    
    try:
        # =================================================================
        # STEP 1: SEARCH STRATEGY GENERATION
        # =================================================================
        
        print(f"\n{'='*70}")
        print("📋 STEP 1: GENERATING SEARCH STRATEGY")
        print('='*70)
        
        strategy = keyword_extractor.generate_search_strategy(research_query)
        print(f"✅ Search strategy generated")
        print(f"  🎯 Primary queries: {len(strategy['search_phases']['primary'])}")
        print(f"  🧠 Complexity score: {strategy['complexity_score']:.2f}/3.0")
        print(f"  📊 Intent: {strategy['search_intent']['primary_intent']}")
        
        # =================================================================
        # STEP 2: SEARCH EXECUTION
        # =================================================================
        
        print(f"\n{'='*70}")
        print("🔍 STEP 2: EXECUTING SEARCHES WITH GEMINI")
        print('='*70)
        
        search_queries = strategy['search_phases']['primary'][:max_searches]
        search_start = datetime.now()
        search_results = await search_engine.search_multiple_queries(search_queries)
        search_duration = (datetime.now() - search_start).total_seconds()
        
        print(f"✅ Search completed in {search_duration:.1f} seconds")
        print(f"  📊 Results found: {len(search_results)}")
        unique_domains = len(set(r.url.split('/')[2] for r in search_results if r.url))
        print(f"  🌐 Unique domains: {unique_domains}")
        print(f"  💰 Estimated cost: ~${max_searches * 0.035:.3f}")
        
        # =================================================================
        # STEP 3: CONTENT EXTRACTION
        # =================================================================
        
        print(f"\n{'='*70}")
        print("🕸️  STEP 3: EXTRACTING WEB CONTENT")
        print('='*70)
        
        # Select top results for extraction
        top_results = search_results[:max_crawls * 2]  # Extract from top results
        extraction_start = datetime.now()
        content_results = await content_extractor.extract_content_with_strategy(top_results)
        extraction_duration = (datetime.now() - extraction_start).total_seconds()
        
        crawled_content = content_results['crawled_content']
        successful_content = [c for c in crawled_content if c.success]
        
        print(f"✅ Content extraction completed in {extraction_duration:.1f} seconds")
        print(f"  📄 Successful extractions: {len(successful_content)}")
        print(f"  📏 Total content: {content_results['total_content_length']:,} characters")
        print(f"  🔢 Total tokens: {content_results['total_token_count']:,}")
        
        if not successful_content:
            print("❌ No content extracted - cannot continue with AI polishing")
            return False
        
        # =================================================================
        # STEP 4: AI CONTENT POLISHING (if enabled)
        # =================================================================
        
        polished_content = None
        polish_duration = 0
        
        if polish_level and ai_polisher:
            print(f"\n{'='*70}")
            print("🤖 STEP 4: AI CONTENT POLISHING & REPORT GENERATION")
            print('='*70)
            
            print(f"Starting AI polishing with Gemini {config.ai_polish_model}...")
            print(f"Processing {len(successful_content)} content sources...")
            
            polish_start = datetime.now()
            polished_content = await ai_polisher.polish_content(
                research_query, search_results, crawled_content
            )
            polish_duration = (datetime.now() - polish_start).total_seconds()
            
            if not polished_content:
                print("❌ AI polishing failed")
                return False
            
            print(f"✅ AI polishing completed in {polish_duration:.1f} seconds")
            
            # Analyze polished content
            polished_length = len(polished_content)
            polished_tokens = tokenizer.count_tokens(polished_content) if tokenizer else polished_length // 4
            
            print(f"  📊 Polished length: {polished_length:,} characters")
            print(f"  🔢 Polished tokens: {polished_tokens:,}")
            improvement_ratio = polished_length / content_results['total_content_length']
            print(f"  📈 Improvement ratio: {improvement_ratio:.2f}x")
        
        # =================================================================
        # RESULTS ANALYSIS & PREVIEW
        # =================================================================
        
        total_duration = search_duration + extraction_duration + polish_duration
        final_content = polished_content or "\\n\\n".join([c.content for c in successful_content if c.success])
        
        print(f"\n{'='*70}")
        print("📊 COMPREHENSIVE DEMO RESULTS")
        print('='*70)
        
        print(f"🔍 Search Phase:")
        print(f"   Queries executed: {len(search_queries)}")
        print(f"   Results found: {len(search_results)}")
        print(f"   Duration: {search_duration:.1f}s")
        
        print(f"\\n🕸️  Extraction Phase:")
        print(f"   URLs crawled: {len(top_results)}")
        print(f"   Successful: {len(successful_content)}")
        print(f"   Duration: {extraction_duration:.1f}s")
        
        if polish_level:
            print(f"\\n🤖 AI Polishing Phase:")
            print(f"   Model used: {config.ai_polish_model}")
            print(f"   Enhancement: {'Comprehensive Report' if polish_level == 2 else 'Basic Polish'}")
            print(f"   Duration: {polish_duration:.1f}s")
        
        print(f"\\n⏱️  Total Duration: {total_duration:.1f} seconds")
        print(f"💰 Total Cost: ~${max_searches * 0.035:.3f}")
        
        # =================================================================
        # CONTENT PREVIEW
        # =================================================================
        
        if final_content:
            print(f"\n📄 {'AI-POLISHED' if polish_level else 'RAW'} CONTENT PREVIEW:")
            print("=" * 70)
            
            # Show first 1000 characters as preview
            preview_text = final_content[:1000]
            if len(final_content) > 1000:
                preview_text += f"\\n\\n... [FULL CONTENT: {len(final_content):,} CHARACTERS] ..."
            
            print(preview_text)
            print("=" * 70)
        
        # =================================================================
        # SAVE RESULTS
        # =================================================================
        
        # Create results directory
        os.makedirs('demo_results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save comprehensive results
        demo_results = {
            'demo_type': 'ai_polishing_comprehensive',
            'timestamp': datetime.now().isoformat(),
            'research_query': research_query,
            'configuration': {
                'max_searches': max_searches,
                'max_crawls': max_crawls,
                'ai_polish_level': polish_level,
                'ai_polish_model': config.ai_polish_model if polish_level else None
            },
            'performance_metrics': {
                'search_duration': search_duration,
                'extraction_duration': extraction_duration,
                'polish_duration': polish_duration,
                'total_duration': total_duration,
                'estimated_cost': max_searches * 0.035
            },
            'results_summary': {
                'search_results_count': len(search_results),
                'successful_extractions': len(successful_content),
                'raw_content_length': content_results['total_content_length'],
                'raw_token_count': content_results['total_token_count'],
                'final_content_length': len(final_content),
                'improvement_ratio': len(final_content) / content_results['total_content_length'] if content_results['total_content_length'] > 0 else 1.0
            },
            'final_content': final_content
        }
        
        result_file = f"demo_results/ai_polish_demo_{timestamp}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(demo_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 Complete demo results saved: {result_file}")
        
        # Save clean report.md file if AI polished
        if polished_content:
            # Save the main report.md file that users actually want  
            report_file = "demo_results/report.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(f"# {research_query}\\n\\n")
                f.write(f"*Research report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\\n\\n")
                f.write(polished_content)
            
            print(f"📄 **MAIN REPORT SAVED: {report_file}**")
            
            # Also save timestamped version for demo archival
            timestamped_file = f"demo_results/ai_polish_demo_{timestamp}.md"
            with open(timestamped_file, 'w', encoding='utf-8') as f:
                f.write(f"# {research_query}\\n\\n")
                f.write(f"*Demo report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\\n\\n")
                f.write(polished_content)
            
            print(f"📄 Demo archived copy saved: {timestamped_file}")
        
        # =================================================================
        # FINAL SUCCESS SUMMARY
        # =================================================================
        
        print(f"\n{'='*70}")
        print("🎉 AI POLISHING DEMO COMPLETED SUCCESSFULLY!")
        print(f"✅ Research query processed: '{research_query}'")
        print(f"✅ {len(search_results)} search results found and analyzed")
        print(f"✅ {len(successful_content)} websites successfully extracted")
        if polish_level:
            print(f"✅ Content {'comprehensively enhanced' if polish_level == 2 else 'polished'} with AI")
            print(f"✅ Professional research report generated")
        print(f"✅ Complete workflow executed in {total_duration:.1f} seconds")
        print(f"✅ Total cost: ~${max_searches * 0.035:.3f}")
        print('='*70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ AI POLISHING DEMO FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main demo function."""
    try:
        success = await run_ai_polishing_demo()
        
        if success:
            print(f"\n🚀 The Gemini Deep Research Agent is fully operational!")
            print("Features demonstrated:")
            print("✓ Intelligent keyword extraction and search strategy")
            print("✓ Google Search with Gemini Grounding") 
            print("✓ Clean web content extraction with paywall detection")
            print("✓ AI-powered content polishing and report generation")
            print("✓ Professional blog-ready output")
            print("✓ Cost tracking and performance optimization")
            
            print(f"\n📚 Ready to use in your projects:")
            print(f"   See: examples/usage_example.py")
        else:
            print(f"\n⚠️  Demo needs attention - check configuration and dependencies")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(main())