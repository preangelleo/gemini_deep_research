#!/usr/bin/env python3
"""
GEMINI SEARCH ENGINE - INTERACTIVE DEMO
=======================================

Test the Google Search functionality with Gemini Grounding.
This demo allows you to:
- Test search queries interactively
- See detailed search results with relevance scores
- Understand token usage and costs
- Save results for analysis

WHAT TO EXPECT:
- Input any search query
- Get 5-10 high-quality search results
- See relevance scores and source types
- Automatic cost tracking (~$0.035 per search)
- Results saved with crash recovery

USAGE:
python examples/interactive_search_demo.py
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

from deep_research_agent import GeminiSearchEngine, KeywordExtractor, ResearchConfig, TokenCounter

def save_results_to_file(query: str, results: list, summary: dict):
    """Save search results to file with crash recovery."""
    os.makedirs('demo_results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"demo_results/search_demo_{timestamp}.json"
    
    data = {
        'demo_type': 'interactive_search',
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'results_count': len(results),
        'results': [result.to_dict() for result in results],
        'summary': summary
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return filename
    except Exception as e:
        print(f"Warning: Could not save results - {e}")
        return None

def get_user_input():
    """Get search query from user with helpful prompts."""
    print("🔍 GEMINI SEARCH ENGINE - INTERACTIVE DEMO")
    print("=" * 60)
    print("Test Google Search with Gemini Grounding technology")
    print("=" * 60)
    
    # Example queries
    examples = [
        "Latest AI developments 2024",
        "Climate change solutions renewable energy", 
        "Space exploration Mars missions",
        "Quantum computing breakthrough news",
        "Sustainable technology innovations"
    ]
    
    print(f"\n💡 EXAMPLE QUERIES:")
    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example}")
    
    # Get user input
    user_query = input(f"\n📝 Enter your search query (or number 1-{len(examples)}): ").strip()
    
    # Handle numbered selection
    if user_query.isdigit():
        num = int(user_query)
        if 1 <= num <= len(examples):
            query = examples[num - 1]
            print(f"✓ Selected: {query}")
        else:
            query = examples[0]  # Default
            print(f"✓ Invalid number, using default: {query}")
    elif user_query:
        query = user_query
        print(f"✓ Custom query: {query}")
    else:
        query = examples[0]  # Default
        print(f"✓ Using default: {query}")
    
    return query

async def run_interactive_search_demo():
    """Run the interactive search demonstration."""
    
    # Get user input
    search_query = get_user_input()
    
    print(f"\n🔧 INITIALIZING SEARCH ENGINE...")
    
    # Initialize components
    config = ResearchConfig.from_env()
    config.debug_mode = True
    config.max_searches_per_task = 1  # Single search for demo
    
    if not config.gemini_api_key:
        print("❌ ERROR: GEMINI_API_KEY not found in .env file")
        return False
    
    tokenizer = TokenCounter(config)
    keyword_extractor = KeywordExtractor(config)
    search_engine = GeminiSearchEngine(config, tokenizer)
    
    if not search_engine.client:
        print("❌ ERROR: Failed to initialize Gemini Search Engine")
        return False
    
    print("✅ Search engine initialized successfully")
    print(f"🔍 Model: {config.gemini_model}")
    print(f"💰 Cost per search: ~$0.035")
    
    try:
        print(f"\n{'='*60}")
        print("🔍 EXECUTING SEARCH...")
        print(f"Query: {search_query}")
        print('='*60)
        
        start_time = datetime.now()
        
        # Execute the search
        search_results = await search_engine.search_with_grounding(
            search_query, 
            max_results=config.search_results_per_query
        )
        
        end_time = datetime.now()
        search_duration = (end_time - start_time).total_seconds()
        
        if not search_results:
            print("❌ No search results returned")
            return False
        
        print(f"✅ Search completed in {search_duration:.1f} seconds")
        print(f"📊 Found {len(search_results)} results")
        
        # Analyze results
        print(f"\n{'='*60}")
        print("📊 SEARCH RESULTS ANALYSIS")
        print('='*60)
        
        domains = set()
        source_types = {}
        avg_relevance = 0
        
        for result in search_results:
            # Extract domain
            try:
                domain = result.url.split('/')[2] if result.url else 'unknown'
                domains.add(domain)
            except:
                domains.add('unknown')
            
            # Count source types
            source_types[result.source_type] = source_types.get(result.source_type, 0) + 1
            avg_relevance += result.relevance_score
        
        avg_relevance = avg_relevance / len(search_results) if search_results else 0
        
        print(f"🌐 Unique domains: {len(domains)}")
        print(f"📈 Average relevance: {avg_relevance:.2f}/3.0")
        print(f"📑 Source types: {dict(source_types)}")
        
        # Show detailed results
        print(f"\n{'='*60}")
        print("🔍 DETAILED SEARCH RESULTS")
        print('='*60)
        
        for i, result in enumerate(search_results, 1):
            print(f"\n{i}. 📄 {result.title}")
            print(f"   🔗 {result.url}")
            print(f"   📊 Relevance: {result.relevance_score:.2f}/3.0")
            print(f"   🏷️  Type: {result.source_type}")
            if hasattr(result, 'snippet') and result.snippet:
                snippet = result.snippet[:200] + "..." if len(result.snippet) > 200 else result.snippet
                print(f"   📝 Snippet: {snippet}")
            print(f"   ─" * 50)
        
        # Create summary
        summary = {
            'search_duration': search_duration,
            'results_count': len(search_results),
            'unique_domains': len(domains),
            'average_relevance': avg_relevance,
            'source_types': source_types,
            'estimated_cost': 0.035  # $0.035 per search
        }
        
        # Save results
        filename = save_results_to_file(search_query, search_results, summary)
        if filename:
            print(f"\n📁 Results saved to: {filename}")
        
        # Final summary
        print(f"\n{'='*60}")
        print("🎉 SEARCH DEMO COMPLETED SUCCESSFULLY!")
        print(f"✅ Query processed: '{search_query}'")
        print(f"✅ Results found: {len(search_results)} high-quality sources")
        print(f"✅ Processing time: {search_duration:.1f} seconds")
        print(f"✅ Estimated cost: $0.035")
        print('='*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ SEARCH DEMO FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main demo function."""
    try:
        success = await run_interactive_search_demo()
        
        if success:
            print(f"\n🚀 Ready to try web content extraction?")
            print(f"   Run: python examples/interactive_crawler_demo.py")
            print(f"\n🚀 Ready for full AI research?")
            print(f"   Run: python examples/usage_example.py")
        else:
            print(f"\n⚠️  Demo needs attention - check your .env configuration")
    
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(main())