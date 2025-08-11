#!/usr/bin/env python3
"""
WEB CONTENT EXTRACTOR - INTERACTIVE DEMO
========================================

Test the web content extraction with clean markdown generation.
This demo allows you to:
- Test any URL for content extraction quality
- See clean markdown without messy hyperlinks
- Check paywall detection and content analysis
- Verify international content support (Chinese, etc.)

WHAT TO EXPECT:
- Input any URL to test
- Get clean, LLM-ready markdown content
- Content quality analysis and cleanliness grading
- Paywall detection and blocking checks
- Support for various content types (news, articles, research)

USAGE:
python examples/interactive_crawler_demo.py
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
    WebContentExtractor,
    ResearchConfig,
    TokenCounter,
    SearchResult
)

def get_user_input():
    """Get URL and preferences from user with helpful examples."""
    print("🕸️  WEB CONTENT EXTRACTOR - INTERACTIVE DEMO")
    print("=" * 70)
    print("Test clean markdown extraction from any website")
    print("=" * 70)
    
    # Example URLs for different content types
    examples = [
        ("Wikipedia Article", "https://en.wikipedia.org/wiki/Artificial_intelligence"),
        ("News Article", "https://www.reuters.com/technology/"),  
        ("Technical Blog", "https://openai.com/blog/"),
        ("Research Paper", "https://arxiv.org/abs/2023.12345"),
        ("WeChat Article", "https://mp.weixin.qq.com/s/0dY8Vwg6_gboEmJtG4nYiw")
    ]
    
    print(f"\n💡 EXAMPLE URLS TO TEST:")
    for i, (desc, url) in enumerate(examples, 1):
        print(f"  {i}. {desc}")
        print(f"     {url}")
    
    # Get user input
    user_input = input(f"\n📝 Enter URL or number (1-{len(examples)}): ").strip()
    
    # Handle numbered selection or URL
    if user_input.isdigit():
        num = int(user_input)
        if 1 <= num <= len(examples):
            url = examples[num - 1][1]
            print(f"✓ Selected: {examples[num - 1][0]}")
            print(f"   URL: {url}")
        else:
            url = examples[0][1]  # Default
            print(f"✓ Invalid number, using default: Wikipedia AI")
    elif user_input.startswith('http'):
        url = user_input
        print(f"✓ Custom URL: {url}")
    else:
        url = examples[0][1]  # Default
        print(f"✓ Using default: Wikipedia AI article")
    
    # Get preview preferences
    print(f"\n📄 CONTENT PREVIEW OPTIONS:")
    print(f"1. Short preview (500 chars) - Quick overview")
    print(f"2. Medium preview (1500 chars) - Detailed sample") 
    print(f"3. Full content - Complete extracted text")
    
    preview_choice = input("Choose preview length (1-3, default=2): ").strip()
    if preview_choice == "1":
        preview_length = 500
    elif preview_choice == "3":
        preview_length = -1  # Full content
    else:
        preview_length = 1500  # Medium (default)
    
    # Save results option
    save_results = input("\\n💾 Save results to file? (y/n, default=y): ").strip().lower()
    save_to_file = save_results != 'n'
    
    return url, preview_length, save_to_file

async def run_url_extraction_demo(url: str, preview_length: int = 1500, save_to_file: bool = True):
    """Demonstrate content extraction from a specific URL."""
    
    print(f"\n🔧 INITIALIZING WEB EXTRACTOR...")
    
    # Initialize components
    config = ResearchConfig.from_env()
    config.debug_mode = True
    config.max_concurrent_crawls = 1  # Single URL for demo
    
    tokenizer = TokenCounter(config)
    extractor = WebContentExtractor(config, tokenizer)
    
    if not extractor.crawler_initialized:
        print("❌ ERROR: WebContentExtractor not initialized")
        print("Run: pip install crawl4ai && playwright install")
        return False
    
    print("✅ Web content extractor initialized")
    print(f"🧹 Clean markdown generation enabled")
    
    # Show configuration
    blacklist_count = len(extractor.config.paywall_url_blacklist) if extractor.config.paywall_url_blacklist else 0
    print(f"🚫 Paywall blacklist: {blacklist_count} URL patterns loaded")
    
    # Check if URL is blacklisted
    is_blacklisted = extractor._is_url_in_paywall_blacklist(url)
    if is_blacklisted:
        print(f"⚠️  WARNING: URL matches paywall blacklist!")
        print(f"   This URL is normally skipped during research.")
        proceed = input("Continue testing anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Demo cancelled.")
            return False
    
    # Auto-detect content type
    if 'mp.weixin.qq.com' in url:
        source_type = 'article'
        print("🔍 Detected: WeChat article (Chinese content)")
    elif 'wikipedia.org' in url:
        source_type = 'documentation'  
        print("🔍 Detected: Wikipedia article")
    elif any(domain in url for domain in ['arxiv.org', 'researchgate.net']):
        source_type = 'research_paper'
        print("🔍 Detected: Research paper")
    elif any(domain in url for domain in ['techcrunch.com', 'reuters.com', 'bbc.com']):
        source_type = 'news'
        print("🔍 Detected: News article")
    else:
        source_type = 'article'
        print("🔍 Detected: Generic article")
    
    # Create test search result
    test_result = SearchResult(
        query="demo_test",
        url=url,
        title=f"Demo Test - {url.split('//')[-1][:50]}",
        source_type=source_type,
        relevance_score=1.0
    )
    
    try:
        print(f"\n{'='*70}")
        print("⏳ EXTRACTING CONTENT...")
        print(f"URL: {url}")
        print(f"Type: {source_type}")
        print(f"This may take 10-30 seconds...")
        print('='*70)
        
        start_time = datetime.now()
        
        # Extract content
        crawled_content = await extractor.extract_content_from_urls([test_result])
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Check extraction success
        if not crawled_content or not crawled_content[0].success:
            print(f"❌ EXTRACTION FAILED")
            if crawled_content and crawled_content[0].error_message:
                print(f"Error: {crawled_content[0].error_message}")
            return False
        
        content = crawled_content[0]
        content_text = content.content
        
        print(f"✅ EXTRACTION SUCCESSFUL!")
        print(f"⏱️  Time: {duration:.1f} seconds")
        
        # =================================================================
        # CONTENT QUALITY ANALYSIS
        # =================================================================
        
        import re
        
        # Count cleanliness metrics
        markdown_links = len(re.findall(r'\\[([^\\]]+)\\]\\([^)]+\\)', content_text))
        html_tags = len(re.findall(r'<[^>]+>', content_text))
        urls = len(re.findall(r'https?://[^\\s]+', content_text))
        
        # Count content structure
        paragraphs = len([p for p in content_text.split('\\n\\n') if len(p.strip()) > 50])
        sentences = len(re.findall(r'[.!?]+', content_text))
        chinese_chars = len(re.findall(r'[\\u4e00-\\u9fff]', content_text))
        
        print(f"\n📊 CONTENT ANALYSIS:")
        print(f"───────────────────────────────────")
        print(f"📏 Length: {content.content_length:,} characters")
        print(f"🔢 Tokens: {content.token_count:,} tokens") 
        print(f"📄 Paragraphs: {paragraphs}")
        print(f"✏️  Sentences: {sentences}")
        if chinese_chars > 0:
            print(f"🇨🇳 Chinese chars: {chinese_chars:,}")
        
        print(f"\n🧹 CLEANLINESS ASSESSMENT:")
        print(f"───────────────────────────────────")
        print(f"🔗 Markdown links: {markdown_links}")
        print(f"🏷️  HTML tags: {html_tags}")
        print(f"🌐 URLs: {urls}")
        
        total_messy = markdown_links + html_tags + urls
        if total_messy == 0:
            cleanliness_grade = "A+"
            print(f"✅ PERFECT! No messy formatting")
        elif total_messy <= 5:
            cleanliness_grade = "A"
            print(f"✅ EXCELLENT! Only {total_messy} minor issues")
        elif total_messy <= 15:
            cleanliness_grade = "B"
            print(f"⚠️  GOOD. {total_messy} issues detected")
        else:
            cleanliness_grade = "C"
            print(f"❌ NEEDS WORK. {total_messy} formatting problems")
        
        # Paywall detection test
        print(f"\n🔒 PAYWALL DETECTION:")
        print(f"───────────────────────────────────")
        is_blocked = extractor._is_paywall_or_blocked(content_text)
        min_length = extractor.config.min_legitimate_content_length
        
        if is_blocked:
            paywall_status = "🚫 BLOCKED/PAYWALL"
            print(f"❌ Content appears blocked or paywalled")
            print(f"   Length: {content.content_length} < {min_length} minimum")
        else:
            paywall_status = "✅ ACCESSIBLE"
            print(f"✅ Legitimate content detected")
            print(f"   Length: {content.content_length} >= {min_length}")
        
        # Overall quality assessment
        if paragraphs >= 3 and sentences >= 10:
            content_quality = "🎯 HIGH QUALITY"
        elif paragraphs >= 2 and sentences >= 5:
            content_quality = "📝 MODERATE QUALITY"
        else:
            content_quality = "📄 LIMITED QUALITY"
        
        print(f"\n🏆 OVERALL ASSESSMENT:")
        print(f"───────────────────────────────────")
        print(f"🧹 Cleanliness: {cleanliness_grade}")
        print(f"📊 Content Quality: {content_quality}")
        print(f"🔓 Access Status: {paywall_status}")
        
        # =================================================================
        # CONTENT PREVIEW
        # =================================================================
        
        print(f"\n📄 EXTRACTED CONTENT PREVIEW:")
        print("=" * 70)
        
        if preview_length == -1:
            # Show full content
            print(content_text)
        else:
            # Show preview
            preview_text = content_text[:preview_length]
            if len(content_text) > preview_length:
                preview_text += f"\n\n... [FULL CONTENT: {len(content_text):,} chars] ..."
            print(preview_text)
        
        print("=" * 70)
        
        # =================================================================
        # SAVE RESULTS
        # =================================================================
        
        if save_to_file:
            os.makedirs('demo_results', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            demo_results = {
                'demo_type': 'interactive_crawler',
                'timestamp': datetime.now().isoformat(),
                'url': url,
                'source_type': source_type,
                'extraction_stats': {
                    'success': content.success,
                    'content_length': content.content_length,
                    'token_count': content.token_count,
                    'extraction_time': duration
                },
                'quality_analysis': {
                    'paragraphs': paragraphs,
                    'sentences': sentences,
                    'chinese_characters': chinese_chars,
                    'cleanliness_grade': cleanliness_grade,
                    'content_quality': content_quality,
                    'paywall_status': paywall_status,
                    'messy_elements': total_messy
                },
                'full_content': content_text
            }
            
            result_file = f"demo_results/crawler_demo_{timestamp}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(demo_results, f, indent=2, ensure_ascii=False)
            
            print(f"\n📁 Demo results saved to: {result_file}")
        
        # Success evaluation
        success = (
            content.success and 
            total_messy <= 10 and 
            paragraphs >= 2 and
            content.content_length >= 200 and
            not is_blocked
        )
        
        print(f"\n{'='*70}")
        if success:
            print("🎉 EXTRACTION DEMO - SUCCESS!")
            print("✅ High-quality, clean content extracted")
            print("✅ Ready for LLM processing")
        else:
            print("⚠️  EXTRACTION DEMO - MIXED RESULTS")
            print("Content extracted but may need refinement")
        print('='*70)
        
        return success
        
    except Exception as e:
        print(f"\n❌ EXTRACTION ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main demo function."""
    try:
        # Get user preferences
        url, preview_length, save_to_file = get_user_input()
        
        # Run extraction demo
        success = await run_url_extraction_demo(url, preview_length, save_to_file)
        
        if success:
            print(f"\n🚀 Ready to try full AI research with content polishing?")
            print(f"   Run: python examples/usage_example.py")
        else:
            print(f"\n⚠️  Demo needs attention - try a different URL or check dependencies")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(main())