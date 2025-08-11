#!/usr/bin/env python3
"""
GEMINI DEEP RESEARCH AGENT - USAGE EXAMPLE
==========================================

This example demonstrates how to use the Gemini Deep Research Agent
to perform comprehensive research on any topic with AI-powered content polishing.

FEATURES:
- Intelligent keyword extraction and search strategy
- Google Search with Gemini Grounding
- Clean web content extraction with Crawl4AI 
- Paywall detection and avoidance
- AI-powered content polishing with Gemini 2.5 Pro
- Blog-ready markdown output

REQUIREMENTS:
1. Set up your .env file with GEMINI_API_KEY
2. Install dependencies: pip install -r requirements.txt
3. Install Playwright browsers: playwright install

USAGE:
Simply modify the parameters below and run:
python examples/usage_example.py
"""

import sys
import os
import asyncio
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from deep_research_agent import (
    DeepResearchAgent,
    ResearchConfig
)

async def main():
    """
    Main usage example for Gemini Deep Research Agent.
    
    CUSTOMIZE THESE PARAMETERS:
    """
    
    # =================================================================
    # 🔧 RESEARCH CONFIGURATION - CUSTOMIZE AS NEEDED
    # =================================================================
    
    # Your research query - change this to whatever you want to research
    research_query = "AI safety regulations impact on tech companies 2024"
    
    # Research intensity (affects cost and depth)
    max_searches = 5          # Number of search queries (default: 10, cost: ~$0.18)
    max_concurrent_crawls = 3  # Parallel web crawling (default: 10)
    
    # AI Content Polishing Options:
    # False = No polishing (just raw extracted content)
    # True = Basic cleaning and polishing
    # 2 = Comprehensive report generation (blog-ready)
    ai_polish_level = 2
    
    # Output preferences
    save_results = True        # Save results to files
    show_preview = True        # Show content preview in console
    
    # Advanced options (usually don't need to change)
    debug_mode = True          # Enable detailed logging
    enable_paywall_detection = True  # Skip paywalled content
    
    # =================================================================
    # 🚀 RESEARCH EXECUTION - NO CHANGES NEEDED BELOW
    # =================================================================
    
    print("🔬 GEMINI DEEP RESEARCH AGENT")
    print("=" * 50)
    print(f"📋 Research Query: {research_query}")
    print(f"🔍 Max Searches: {max_searches} (~${max_searches * 0.035:.3f})")
    print(f"🤖 AI Polish: {'Comprehensive Report' if ai_polish_level == 2 else 'Basic' if ai_polish_level else 'Disabled'}")
    print("=" * 50)
    
    try:
        # Initialize configuration with custom parameters
        config = ResearchConfig.from_env()
        config.max_searches_per_task = max_searches
        config.max_concurrent_crawls = max_concurrent_crawls
        config.ai_polish_content = ai_polish_level
        config.debug_mode = debug_mode
        
        # Validate API key
        if not config.gemini_api_key:
            print("❌ ERROR: GEMINI_API_KEY not found in .env file")
            print("Please add your Gemini API key to .env file:")
            print("GEMINI_API_KEY=your_api_key_here")
            return
        
        # Initialize the research agent
        print("🔧 Initializing Deep Research Agent...")
        agent = DeepResearchAgent(config)
        
        if not agent.is_initialized():
            print("❌ ERROR: Failed to initialize research agent")
            print("Check your .env configuration and dependencies")
            return
        
        print("✅ Research agent initialized successfully!")
        
        # Execute comprehensive research
        print(f"\n⏳ Starting comprehensive research...")
        print(f"This may take 2-5 minutes depending on complexity...")
        
        start_time = datetime.now()
        
        # Perform the research
        results = await agent.research(research_query)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Check results
        if not results:
            print("❌ Research failed - no results returned")
            return
        
        print(f"\n✅ Research completed in {duration:.1f} seconds!")
        
        # =================================================================
        # 📊 RESULTS ANALYSIS
        # =================================================================
        
        search_results = results.get('search_results', [])
        crawled_content = results.get('crawled_content', [])
        polished_content = results.get('polished_content', '')
        successful_extractions = len([c for c in crawled_content if c.success])
        
        print(f"\n📊 RESEARCH SUMMARY:")
        print(f"  🔍 Search results found: {len(search_results)}")
        print(f"  🕸️  Successful extractions: {successful_extractions}")
        print(f"  🔢 Raw content length: {sum(c.content_length for c in crawled_content if c.success):,} chars")
        
        if polished_content:
            print(f"  🤖 AI polished content: {len(polished_content):,} chars")
            print(f"  ✨ Content enhanced with Gemini {config.ai_polish_model}")
        
        # =================================================================
        # 💾 SAVE RESULTS
        # =================================================================
        
        if save_results:
            # Create results directory
            os.makedirs('research_results', exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_query = "".join(c for c in research_query if c.isalnum() or c in (' ', '-')).strip()
            safe_query = safe_query.replace(' ', '_')[:30]
            
            # Save comprehensive results as JSON
            results_file = f"research_results/{safe_query}_{timestamp}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'query': research_query,
                    'timestamp': datetime.now().isoformat(),
                    'config': {
                        'max_searches': max_searches,
                        'ai_polish_level': ai_polish_level,
                        'model_used': config.ai_polish_model
                    },
                    'results': results,
                    'polished_content': polished_content
                }, f, indent=2, ensure_ascii=False)
            
            print(f"📁 Complete results saved: {results_file}")
            
            # Save clean report.md file (main output for users)
            if polished_content:
                # Save the main report.md file that users actually want
                report_file = "research_results/report.md"
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(f"# {research_query}\n\n")
                    f.write(f"*Research report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                    f.write(polished_content)
                
                print(f"📄 **MAIN REPORT SAVED: {report_file}**")
                
                # Also save timestamped version for archival
                timestamped_file = f"research_results/{safe_query}_{timestamp}.md"
                with open(timestamped_file, 'w', encoding='utf-8') as f:
                    f.write(f"# {research_query}\n\n")
                    f.write(f"*Research report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                    f.write(polished_content)
                
                print(f"📄 Archived copy saved: {timestamped_file}")
        
        # =================================================================
        # 🖥️  PREVIEW RESULTS
        # =================================================================
        
        if show_preview and polished_content:
            print(f"\n📄 RESEARCH REPORT PREVIEW:")
            print("=" * 70)
            
            # Show first 1500 characters as preview
            preview = polished_content[:1500]
            if len(polished_content) > 1500:
                preview += f"\n\n... [FULL REPORT IS {len(polished_content):,} CHARACTERS] ..."
            
            print(preview)
            print("=" * 70)
        
        # =================================================================
        # 🎉 SUCCESS SUMMARY
        # =================================================================
        
        print(f"\n🎉 RESEARCH COMPLETED SUCCESSFULLY!")
        print(f"✅ Query analyzed and search strategy generated")
        print(f"✅ {len(search_results)} search results found via Gemini")
        print(f"✅ {successful_extractions} websites successfully crawled")
        
        if polished_content:
            print(f"✅ Content polished and enhanced with AI")
            print(f"✅ Blog-ready research report generated")
        
        if save_results:
            print(f"✅ Results saved to research_results/ directory")
        
        print(f"\n💰 Estimated cost: ~${max_searches * 0.035:.3f} (Google Search)")
        print(f"⏱️  Total time: {duration:.1f} seconds")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Research interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

# =================================================================
# 🚀 QUICK START EXAMPLES
# =================================================================

def example_basic_research():
    """Quick example for basic research without AI polishing."""
    return {
        'research_query': "Latest developments in quantum computing 2024",
        'max_searches': 3,
        'ai_polish_level': False,  # No AI polishing
        'max_concurrent_crawls': 2
    }

def example_comprehensive_research():
    """Example for comprehensive research with full AI polishing."""
    return {
        'research_query': "Climate change impact on global economy 2024",
        'max_searches': 10,
        'ai_polish_level': 2,  # Comprehensive report generation
        'max_concurrent_crawls': 5
    }

def example_budget_research():
    """Low-cost example for budget-conscious research."""
    return {
        'research_query': "Best practices for remote work productivity",
        'max_searches': 2,  # Only ~$0.07 cost
        'ai_polish_level': True,  # Basic polishing
        'max_concurrent_crawls': 2
    }

if __name__ == "__main__":
    """
    Run this script to start your research!
    
    To use different examples, uncomment one of these:
    """
    
    # Use default parameters from main()
    asyncio.run(main())
    
    # Or try one of the preset examples:
    # config = example_basic_research()
    # asyncio.run(main(**config))