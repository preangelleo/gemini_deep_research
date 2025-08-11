"""
Gemini Deep Research Agent - Main Module
========================================

A custom implementation of Google's Deep Research functionality using:
- Gemini API with Search Grounding
- Crawl4AI for web scraping
- Async/parallel processing for optimal performance

Author: Deep Research Team
Version: 1.0.0
"""

import os
import re
import asyncio
import logging
import sys
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add prompts directory to Python path for system prompt imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts'))

# Third-party imports
try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

try:
    from crawl4ai import AsyncWebCrawler
except ImportError:
    AsyncWebCrawler = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import system prompts for AI content polishing
try:
    from research_analysis_reporter import DEFAULT_SYSTEM_PROMPT, get_system_prompt
except ImportError:
    logger.warning("Could not import system prompts - using fallback prompts")
    DEFAULT_SYSTEM_PROMPT = None
    get_system_prompt = None


@dataclass
class TokenUsage:
    """Track token usage throughout the research process."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def add_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Add token usage to running totals."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens = self.input_tokens + self.output_tokens
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens, 
            'total_tokens': self.total_tokens,
            'estimated_cost': self.estimated_cost,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ResearchConfig:
    """Configuration settings for the Deep Research Agent."""
    
    # API Configuration
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-pro"
    gemini_flash_model: str = "gemini-2.5-flash"
    
    # Context Window Limits (tokens)
    gemini_input_context_limit: int = 1000000  # 1M tokens for Gemini 2.5
    gemini_output_context_limit: int = 8192    # 8K tokens output limit
    gemini_total_context_limit: int = 1000000  # Total context window
    
    # Search Configuration
    max_searches_per_task: int = 10
    search_results_per_query: int = 10
    enable_google_search_grounding: bool = True
    
    # Crawling Configuration
    max_concurrent_crawls: int = 10
    crawl_timeout_per_url: int = 30
    max_crawl_timeout_minutes: int = 5
    max_content_length_per_url: int = 50000
    
    # Token Management
    reserved_tokens_for_report: int = 100000
    max_total_tokens: int = 800000
    token_safety_buffer: int = 50000
    
    # Performance Settings
    enable_async_processing: bool = True
    enable_parallel_crawling: bool = True
    worker_thread_count: int = 8
    
    # Cost Management
    max_cost_per_task: float = 5.00
    enable_cost_tracking: bool = True
    cost_alert_threshold: float = 10.00
    
    # Content Filtering
    min_content_length: int = 500
    min_legitimate_content_length: int = 800  # Detect paywall/verification pages
    max_content_length: int = 100000
    skip_non_text_content: bool = True
    paywall_url_blacklist: List[str] = None  # URL prefixes to skip
    
    # AI Content Polishing
    ai_polish_content: Union[bool, int] = False  # False, True, or 2 for comprehensive
    ai_polish_model: str = "gemini-2.5-pro"
    max_polish_report_tokens: int = 32000  # Gemini 2.5 Pro max: 64,000
    
    # Output Settings
    default_output_format: str = "markdown"
    include_citations: bool = True
    max_report_length: int = 20000
    
    # Debug Settings
    debug_mode: bool = False
    save_intermediate_results: bool = False
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> 'ResearchConfig':
        """Create configuration from environment variables."""
        return cls(
            # API Configuration
            gemini_api_key=os.getenv('GEMINI_API_KEY', ''),
            gemini_model=os.getenv('GEMINI_MODEL', 'gemini-2.5-pro'),
            gemini_flash_model=os.getenv('GEMINI_FLASH_MODEL', 'gemini-2.5-flash'),
            
            # Context Window Limits
            gemini_input_context_limit=int(os.getenv('GEMINI_INPUT_CONTEXT_LIMIT', 1000000)),
            gemini_output_context_limit=int(os.getenv('GEMINI_OUTPUT_CONTEXT_LIMIT', 8192)),
            gemini_total_context_limit=int(os.getenv('GEMINI_TOTAL_CONTEXT_LIMIT', 1000000)),
            
            # Search Configuration
            max_searches_per_task=int(os.getenv('MAX_SEARCHES_PER_TASK', 10)),
            search_results_per_query=int(os.getenv('SEARCH_RESULTS_PER_QUERY', 10)),
            enable_google_search_grounding=os.getenv('ENABLE_GOOGLE_SEARCH_GROUNDING', 'true').lower() == 'true',
            
            # Crawling Configuration
            max_concurrent_crawls=int(os.getenv('MAX_CONCURRENT_CRAWLS', 10)),
            crawl_timeout_per_url=int(os.getenv('CRAWL_TIMEOUT_PER_URL', 30)),
            max_crawl_timeout_minutes=int(os.getenv('MAX_CRAWL_TIMEOUT_MINUTES', 5)),
            max_content_length_per_url=int(os.getenv('MAX_CONTENT_LENGTH_PER_URL', 50000)),
            
            # Token Management
            reserved_tokens_for_report=int(os.getenv('RESERVED_TOKENS_FOR_REPORT', 100000)),
            max_total_tokens=int(os.getenv('MAX_TOTAL_TOKENS', 800000)),
            token_safety_buffer=int(os.getenv('TOKEN_SAFETY_BUFFER', 50000)),
            
            # Performance Settings
            enable_async_processing=os.getenv('ENABLE_ASYNC_PROCESSING', 'true').lower() == 'true',
            enable_parallel_crawling=os.getenv('ENABLE_PARALLEL_CRAWLING', 'true').lower() == 'true',
            worker_thread_count=int(os.getenv('WORKER_THREAD_COUNT', 8)),
            
            # Cost Management
            max_cost_per_task=float(os.getenv('MAX_COST_PER_TASK', 5.00)),
            enable_cost_tracking=os.getenv('ENABLE_COST_TRACKING', 'true').lower() == 'true',
            cost_alert_threshold=float(os.getenv('COST_ALERT_THRESHOLD', 10.00)),
            
            # Content Filtering
            min_content_length=int(os.getenv('MIN_CONTENT_LENGTH', 500)),
            min_legitimate_content_length=int(os.getenv('MIN_LEGITIMATE_CONTENT_LENGTH', 800)),
            max_content_length=int(os.getenv('MAX_CONTENT_LENGTH', 100000)),
            skip_non_text_content=os.getenv('SKIP_NON_TEXT_CONTENT', 'true').lower() == 'true',
            paywall_url_blacklist=cls._parse_paywall_blacklist(os.getenv('PAYWALL_URL_BLACKLIST', '')),
            
            # AI Content Polishing
            ai_polish_content=cls._parse_polish_setting(os.getenv('AI_POLISH_CONTENT', 'false')),
            ai_polish_model=os.getenv('AI_POLISH_MODEL', 'gemini-2.5-pro'),
            max_polish_report_tokens=int(os.getenv('MAX_POLISH_REPORT_TOKENS', 32000)),
            
            # Output Settings
            default_output_format=os.getenv('DEFAULT_OUTPUT_FORMAT', 'markdown'),
            include_citations=os.getenv('INCLUDE_CITATIONS', 'true').lower() == 'true',
            max_report_length=int(os.getenv('MAX_REPORT_LENGTH', 20000)),
            
            # Debug Settings
            debug_mode=os.getenv('DEBUG_MODE', 'false').lower() == 'true',
            save_intermediate_results=os.getenv('SAVE_INTERMEDIATE_RESULTS', 'false').lower() == 'true',
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
        )
    
    @classmethod
    def _parse_paywall_blacklist(cls, blacklist_string: str) -> List[str]:
        """Parse comma-separated paywall blacklist from environment variable."""
        if not blacklist_string:
            # Comprehensive default paywall URLs - save computing power!
            default_blacklist = [
                # Major News Publications
                'wsj.com', 'nytimes.com', 'ft.com', 'bloomberg.com', 'reuters.com/paywall',
                'economist.com', 'washingtonpost.com', 'newyorker.com', 'theatlantic.com',
                'wired.com', 'harpers.org', 'vanityfair.com', 'nationalreview.com',
                'foreignaffairs.com', 'foreignpolicy.com', 'politico.com/news/', 
                'wsj.com', 'barrons.com', 'marketwatch.com/premium',
                
                # Technology Publications  
                'techcrunch.com/pro', 'theinformation.com', 'stratechery.com',
                'arstechnica.com/civis', 'spectrum.ieee.org/premium',
                
                # Business Publications
                'hbr.org', 'fortune.com/premium', 'inc.com/premium', 'fastcompany.com/premium',
                'mckinsey.com/featured-insights', 'bcg.com/publications',
                
                # Academic & Research
                'nature.com/articles', 'science.org/doi', 'cell.com/action',
                'nejm.org/doi', 'thelancet.com/journals', 'springer.com/article',
                'sciencedirect.com/science/article', 'ieee.org/document',
                'acm.org/doi', 'jstor.org/stable',
                
                # Social Media Paywalls
                'medium.com/@', 'substack.com/', 'patreon.com/posts',
                'onlyfans.com', 'linkedin.com/premium', 'twitter.com/i/premium',
                
                # Chinese Platforms (Often Require Login)
                'mp.weixin.qq.com/s/', 'zhihu.com/question', 'weibo.com/u/',
                'jianshu.com/p/', 'csdn.net/article', 'cnblogs.com/p/',
                
                # Regional News
                'scmp.com/news', 'japantimes.co.jp/news', 'koreatimes.co.kr',
                'timesofindia.indiatimes.com/premium', 'theaustralian.com.au',
                'telegraph.co.uk/premium', 'thetimes.co.uk/article',
                'lemonde.fr/abonnement', 'elpais.com/suscripciones',
                
                # Financial Services
                'seekingalpha.com/premium', 'fool.com/premium', 'zacks.com/premium',
                'morningstar.com/premium', 'valueline.com/premium',
                
                # Legal & Professional
                'law.com/premium', 'americanlawyer.com/premium', 'legaltechnology.com/premium',
                'westlaw.com', 'lexisnexis.com/hottopics',
                
                # Entertainment & Lifestyle
                'netflix.com/watch', 'hulu.com/watch', 'disneyplus.com/movies',
                'spotify.com/premium', 'youtube.com/premium',
                
                # Generic Paywall Patterns
                '/premium/', '/subscription/', '/paywall/', '/member/',
                '/subscriber/', '/pro/', '/plus/', '/unlock/', '/access/'
            ]
            return default_blacklist
        
        # Parse from environment variable
        return [url.strip() for url in blacklist_string.split(',') if url.strip()]
    
    @classmethod
    def _parse_polish_setting(cls, polish_string: str) -> Union[bool, int]:
        """Parse AI polish setting from environment variable."""
        polish_lower = polish_string.lower().strip()
        
        if polish_lower in ['false', '0', 'off', 'no']:
            return False
        elif polish_lower in ['true', '1', 'on', 'yes']:
            return True
        elif polish_lower == '2':
            return 2
        else:
            # Try to parse as integer
            try:
                return int(polish_string)
            except ValueError:
                return False


class KeywordExtractor:
    """
    Intelligent keyword extraction and search query generation.
    
    This class analyzes user research queries and generates optimized search terms
    for comprehensive web research using multiple strategies:
    1. Intent analysis and categorization
    2. Entity extraction and expansion
    3. Temporal and contextual keyword generation
    4. Query diversification for comprehensive coverage
    """
    
    def __init__(self, config: ResearchConfig):
        """Initialize the KeywordExtractor with configuration."""
        self.config = config
        self.tokenizer = None  # Will be set when TokenCounter is available
        
        # Common stop words to filter out
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
            'is', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'
        }
        
        # Research intent patterns
        self.intent_patterns = {
            'trends': ['latest', 'recent', 'current', 'new', 'emerging', 'trending', '2024', '2025'],
            'comparison': ['vs', 'versus', 'compare', 'comparison', 'difference', 'better', 'best'],
            'analysis': ['analysis', 'analyze', 'study', 'research', 'investigation', 'examination'],
            'howto': ['how to', 'tutorial', 'guide', 'instructions', 'steps', 'method', 'process'],
            'definition': ['what is', 'definition', 'meaning', 'explanation', 'overview', 'introduction'],
            'market': ['market', 'industry', 'business', 'commercial', 'economic', 'financial'],
            'technical': ['technical', 'architecture', 'implementation', 'development', 'engineering'],
            'regulation': ['regulation', 'policy', 'law', 'compliance', 'legal', 'governance'],
            'impact': ['impact', 'effect', 'consequence', 'influence', 'result', 'outcome']
        }
        
        # Domain-specific keyword expansions
        self.domain_expansions = {
            'ai': ['artificial intelligence', 'machine learning', 'deep learning', 'neural networks', 'LLM', 'AGI'],
            'crypto': ['cryptocurrency', 'blockchain', 'bitcoin', 'ethereum', 'defi', 'nft'],
            'health': ['healthcare', 'medical', 'pharmaceutical', 'biotechnology', 'clinical'],
            'tech': ['technology', 'software', 'hardware', 'digital', 'innovation', 'startup'],
            'finance': ['financial', 'banking', 'investment', 'trading', 'market', 'economy'],
            'energy': ['renewable energy', 'solar', 'wind', 'battery', 'electric', 'sustainability'],
            'space': ['space exploration', 'satellite', 'rocket', 'aerospace', 'mars', 'moon']
        }
        
        logger.info("KeywordExtractor initialized with pattern matching and domain expansion")
    
    def extract_keywords(self, query: str, max_queries: Optional[int] = None) -> List[str]:
        """
        Extract and generate search keywords from a research query.
        
        Args:
            query: The original research query
            max_queries: Maximum number of search queries to generate
            
        Returns:
            List of optimized search queries
        """
        if max_queries is None:
            max_queries = self.config.max_searches_per_task
        
        # Step 1: Analyze the query intent and extract core concepts
        intent_analysis = self._analyze_intent(query)
        core_keywords = self._extract_core_keywords(query)
        entities = self._extract_entities(query)
        
        # Step 2: Generate base search queries
        base_queries = self._generate_base_queries(query, core_keywords, entities)
        
        # Step 3: Apply query diversification strategies
        diversified_queries = self._diversify_queries(base_queries, intent_analysis)
        
        # Step 4: Add temporal and contextual variations
        enhanced_queries = self._add_temporal_context(diversified_queries, intent_analysis)
        
        # Step 5: Optimize and rank queries
        final_queries = self._optimize_queries(enhanced_queries, max_queries)
        
        if self.config.debug_mode:
            logger.info(f"Generated {len(final_queries)} search queries from: '{query}'")
            for i, q in enumerate(final_queries, 1):
                logger.debug(f"  {i}. {q}")
        
        return final_queries
    
    def _analyze_intent(self, query: str) -> Dict[str, Any]:
        """Analyze the research intent behind the query."""
        query_lower = query.lower()
        detected_intents = []
        
        # Check for intent patterns
        for intent, patterns in self.intent_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                detected_intents.append(intent)
        
        # Determine primary research type
        primary_intent = detected_intents[0] if detected_intents else 'general'
        
        # Extract temporal indicators
        temporal_indicators = []
        current_year = datetime.now().year
        for year in range(current_year - 2, current_year + 2):
            if str(year) in query:
                temporal_indicators.append(str(year))
        
        # Check for recency requirements
        recency_keywords = ['latest', 'recent', 'current', 'new', 'updated']
        needs_recent = any(keyword in query_lower for keyword in recency_keywords)
        
        return {
            'detected_intents': detected_intents,
            'primary_intent': primary_intent,
            'temporal_indicators': temporal_indicators,
            'needs_recent': needs_recent,
            'query_complexity': len(query.split()),
            'has_technical_terms': self._has_technical_terms(query)
        }
    
    def _extract_core_keywords(self, query: str) -> List[str]:
        """Extract core keywords from the query."""
        # Clean and tokenize
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter stop words and short words
        keywords = [
            word for word in words 
            if word not in self.stop_words and len(word) > 2
        ]
        
        # Extract multi-word phrases (bigrams and trigrams)
        phrases = []
        words_clean = query.lower().split()
        
        # Bigrams
        for i in range(len(words_clean) - 1):
            phrase = f"{words_clean[i]} {words_clean[i+1]}"
            if not any(stop in phrase for stop in ['the ', 'a ', 'an ']):
                phrases.append(phrase)
        
        # Trigrams for technical terms
        for i in range(len(words_clean) - 2):
            phrase = f"{words_clean[i]} {words_clean[i+1]} {words_clean[i+2]}"
            if not any(stop in phrase for stop in ['the ', 'a ', 'an ', 'of ', 'in ']):
                phrases.append(phrase)
        
        return keywords + phrases
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract named entities and categorize them."""
        entities = {
            'organizations': [],
            'technologies': [],
            'locations': [],
            'people': [],
            'dates': [],
            'products': []
        }
        
        query_lower = query.lower()
        
        # Common organization patterns
        org_patterns = [r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', r'\b[A-Z]{2,}\b']
        for pattern in org_patterns:
            matches = re.findall(pattern, query)
            entities['organizations'].extend(matches)
        
        # Technology terms (simple pattern matching)
        tech_terms = ['api', 'ai', 'ml', 'blockchain', 'cloud', 'saas', 'sdk', 'iot']
        for term in tech_terms:
            if term in query_lower:
                entities['technologies'].append(term.upper())
        
        # Year extraction
        year_pattern = r'\b(20\d{2}|19\d{2})\b'
        years = re.findall(year_pattern, query)
        entities['dates'].extend(years)
        
        return entities
    
    def _generate_base_queries(self, original_query: str, keywords: List[str], 
                              entities: Dict[str, List[str]]) -> List[str]:
        """Generate base search queries using different strategies."""
        queries = []
        
        # Strategy 1: Use original query as-is
        queries.append(original_query)
        
        # Strategy 2: Core keyword combinations
        if len(keywords) >= 2:
            # Take top keywords and create combinations
            top_keywords = keywords[:4]
            for i in range(len(top_keywords)):
                for j in range(i + 1, min(i + 3, len(top_keywords))):
                    query = f"{top_keywords[i]} {top_keywords[j]}"
                    queries.append(query)
        
        # Strategy 3: Entity-focused queries
        for entity_type, entity_list in entities.items():
            if entity_list and entity_type in ['organizations', 'technologies', 'products']:
                for entity in entity_list[:2]:  # Top 2 entities per type
                    # Combine entity with main keywords
                    if keywords:
                        query = f"{entity} {keywords[0]}"
                        queries.append(query)
        
        # Strategy 4: Long-tail variations
        if len(keywords) >= 3:
            long_tail = " ".join(keywords[:3])
            queries.append(long_tail)
        
        return queries
    
    def _diversify_queries(self, base_queries: List[str], intent_analysis: Dict[str, Any]) -> List[str]:
        """Apply diversification strategies based on intent analysis."""
        diversified = base_queries.copy()
        primary_intent = intent_analysis['primary_intent']
        
        # Intent-based query variations
        intent_modifiers = {
            'trends': ['latest', 'recent trends in', 'emerging', '2024'],
            'comparison': ['compare', 'vs', 'difference between', 'best'],
            'analysis': ['analysis of', 'research on', 'study about', 'report'],
            'howto': ['how to', 'guide to', 'tutorial', 'steps to'],
            'definition': ['what is', 'definition', 'explanation of', 'overview'],
            'market': ['market analysis', 'industry report', 'market size', 'growth'],
            'technical': ['technical overview', 'architecture', 'implementation', 'specs'],
            'regulation': ['regulations', 'compliance', 'policy', 'legal framework'],
            'impact': ['impact of', 'effects', 'consequences', 'influence']
        }
        
        if primary_intent in intent_modifiers:
            modifiers = intent_modifiers[primary_intent]
            for base_query in base_queries[:3]:  # Apply to top 3 base queries
                for modifier in modifiers[:2]:  # Use top 2 modifiers
                    modified_query = f"{modifier} {base_query}"
                    diversified.append(modified_query)
        
        # Domain-specific expansion
        query_text = " ".join(base_queries).lower()
        for domain, expansions in self.domain_expansions.items():
            if domain in query_text:
                for expansion in expansions[:2]:
                    if expansion not in query_text:
                        # Create query with domain expansion
                        expansion_query = f"{base_queries[0]} {expansion}"
                        diversified.append(expansion_query)
        
        return diversified
    
    def _add_temporal_context(self, queries: List[str], intent_analysis: Dict[str, Any]) -> List[str]:
        """Add temporal context to queries when appropriate."""
        enhanced = queries.copy()
        
        # Add current year for trend-related queries
        if intent_analysis['needs_recent'] or 'trends' in intent_analysis['detected_intents']:
            current_year = datetime.now().year
            for query in queries[:5]:  # Apply to top 5 queries
                if str(current_year) not in query and str(current_year - 1) not in query:
                    enhanced.append(f"{query} {current_year}")
                    enhanced.append(f"{query} {current_year - 1}")
        
        # Add "recent" or "latest" for time-sensitive queries
        if intent_analysis['needs_recent']:
            for query in queries[:3]:
                if not any(word in query.lower() for word in ['recent', 'latest', 'current']):
                    enhanced.append(f"recent {query}")
                    enhanced.append(f"latest {query}")
        
        return enhanced
    
    def _optimize_queries(self, queries: List[str], max_queries: int) -> List[str]:
        """Optimize and rank queries for final selection."""
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for query in queries:
            query_clean = query.lower().strip()
            if query_clean not in seen and len(query_clean) > 0:
                seen.add(query_clean)
                unique_queries.append(query.strip())
        
        # Score queries based on multiple factors
        scored_queries = []
        for query in unique_queries:
            score = self._score_query(query)
            scored_queries.append((query, score))
        
        # Sort by score (descending) and take top queries
        scored_queries.sort(key=lambda x: x[1], reverse=True)
        final_queries = [query for query, score in scored_queries[:max_queries]]
        
        return final_queries
    
    def _score_query(self, query: str) -> float:
        """Score a query based on multiple quality factors."""
        score = 0.0
        
        # Base score
        score += 1.0
        
        # Length factor (prefer medium-length queries)
        word_count = len(query.split())
        if 2 <= word_count <= 5:
            score += 2.0
        elif word_count == 1:
            score += 0.5
        elif word_count > 8:
            score -= 1.0
        
        # Specificity bonus (has technical terms or entities)
        if self._has_technical_terms(query):
            score += 1.5
        
        # Recency bonus
        current_year = datetime.now().year
        if any(str(year) in query for year in [current_year, current_year - 1]):
            score += 1.0
        
        # Diversity bonus (contains question words or intent modifiers)
        diversity_terms = ['how', 'what', 'why', 'when', 'compare', 'analysis', 'latest']
        if any(term in query.lower() for term in diversity_terms):
            score += 0.5
        
        return score
    
    def _has_technical_terms(self, text: str) -> bool:
        """Check if text contains technical terminology."""
        technical_indicators = [
            # Technology
            'api', 'sdk', 'framework', 'algorithm', 'protocol', 'architecture',
            # AI/ML
            'artificial intelligence', 'machine learning', 'neural network', 'model',
            # Business/Finance
            'market cap', 'roi', 'kpi', 'revenue', 'valuation', 'funding',
            # Science
            'research', 'study', 'analysis', 'methodology', 'data', 'statistics'
        ]
        
        text_lower = text.lower()
        return any(term in text_lower for term in technical_indicators)
    
    def generate_search_strategy(self, query: str) -> Dict[str, Any]:
        """
        Generate a comprehensive search strategy for a research query.
        
        Returns a strategy including:
        - Primary search queries
        - Search phases (broad to specific)
        - Expected content types
        - Source prioritization
        """
        intent_analysis = self._analyze_intent(query)
        keywords = self.extract_keywords(query)
        
        # Organize queries into search phases
        primary_queries = keywords[:3]
        secondary_queries = keywords[3:7] if len(keywords) > 3 else []
        exploratory_queries = keywords[7:] if len(keywords) > 7 else []
        
        # Determine expected content types
        content_types = ['articles', 'research papers']
        if intent_analysis['primary_intent'] == 'howto':
            content_types.extend(['tutorials', 'guides'])
        elif intent_analysis['primary_intent'] == 'market':
            content_types.extend(['reports', 'market analysis'])
        elif intent_analysis['primary_intent'] == 'technical':
            content_types.extend(['documentation', 'technical specs'])
        
        strategy = {
            'original_query': query,
            'intent_analysis': intent_analysis,
            'search_phases': {
                'primary': primary_queries,
                'secondary': secondary_queries,
                'exploratory': exploratory_queries
            },
            'expected_content_types': content_types,
            'search_priority': self._determine_search_priority(intent_analysis),
            'estimated_searches': len(keywords),
            'complexity_score': self._calculate_complexity_score(query, intent_analysis)
        }
        
        return strategy
    
    def _determine_search_priority(self, intent_analysis: Dict[str, Any]) -> str:
        """Determine search priority based on intent analysis."""
        if intent_analysis['needs_recent']:
            return 'recent_first'
        elif intent_analysis['primary_intent'] in ['definition', 'howto']:
            return 'authoritative_sources'
        elif intent_analysis['primary_intent'] in ['market', 'analysis']:
            return 'comprehensive_coverage'
        else:
            return 'balanced'
    
    def _calculate_complexity_score(self, query: str, intent_analysis: Dict[str, Any]) -> float:
        """Calculate complexity score for research planning."""
        score = 1.0
        
        # Query length factor
        score += len(query.split()) * 0.1
        
        # Intent complexity
        if len(intent_analysis['detected_intents']) > 2:
            score += 0.5
        
        # Technical terms boost
        if intent_analysis['has_technical_terms']:
            score += 0.3
        
        # Multi-faceted research
        complex_intents = ['comparison', 'analysis', 'market']
        if intent_analysis['primary_intent'] in complex_intents:
            score += 0.4
        
        return min(score, 3.0)  # Cap at 3.0


@dataclass
class SearchResult:
    """Represents a search result with metadata."""
    query: str
    url: str = ""
    title: str = ""
    snippet: str = ""
    relevance_score: float = 0.0
    source_type: str = "unknown"  # article, paper, documentation, etc.
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'query': self.query,
            'url': self.url,
            'title': self.title,
            'snippet': self.snippet,
            'relevance_score': self.relevance_score,
            'source_type': self.source_type,
            'timestamp': self.timestamp.isoformat()
        }


class GeminiSearchEngine:
    """
    Google Search integration using Gemini's grounding capabilities.
    
    This class handles:
    1. Search query execution via Gemini API with Google Search grounding
    2. Result parsing and URL extraction
    3. Relevance scoring and ranking
    4. Cost tracking and optimization
    5. Parallel search execution for efficiency
    """
    
    def __init__(self, config: ResearchConfig, tokenizer: Optional['TokenCounter'] = None):
        """Initialize the GeminiSearchEngine."""
        self.config = config
        self.tokenizer = tokenizer
        self.client = None
        self.search_costs = 0.0
        self.search_count = 0
        
        # Initialize Gemini client if API key is available
        if self.config.gemini_api_key:
            try:
                if genai:
                    # Initialize client with API key
                    self.client = genai.Client(api_key=self.config.gemini_api_key)
                    logger.info("GeminiSearchEngine initialized with API key")
                else:
                    logger.error("google-genai library not installed")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
        else:
            logger.warning("No Gemini API key provided")
        
        # Search result cache for avoiding duplicate searches
        self.search_cache = {}
        
        # Cost tracking (Google Search Grounding: $35 per 1000 queries)
        self.cost_per_search = 0.035
        
        logger.info(f"GeminiSearchEngine ready - Max searches per task: {config.max_searches_per_task}")
    
    async def search_multiple_queries(self, queries: List[str], 
                                    max_results_per_query: Optional[int] = None) -> List[SearchResult]:
        """
        Execute multiple search queries in parallel.
        
        Args:
            queries: List of search queries to execute
            max_results_per_query: Maximum results per individual query
            
        Returns:
            List of SearchResult objects
        """
        if not self.client:
            logger.error("Gemini client not initialized - cannot perform searches")
            return []
        
        if max_results_per_query is None:
            max_results_per_query = self.config.search_results_per_query
        
        logger.info(f"Executing {len(queries)} search queries in parallel")
        
        # Execute searches in parallel using asyncio
        if self.config.enable_async_processing:
            tasks = [
                self._execute_single_search(query, max_results_per_query) 
                for query in queries
            ]
            search_results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Sequential execution fallback
            search_results = []
            for query in queries:
                result = await self._execute_single_search(query, max_results_per_query)
                search_results.append(result)
        
        # Flatten results and filter out exceptions
        all_results = []
        for result in search_results:
            if isinstance(result, list):
                all_results.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Search error: {result}")
        
        # Sort by relevance score
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Update cost tracking
        successful_searches = sum(1 for r in search_results if isinstance(r, list))
        self._update_search_costs(successful_searches)
        
        logger.info(f"Search completed: {len(all_results)} results from {successful_searches} queries")
        
        return all_results
    
    async def _execute_single_search(self, query: str, max_results: int) -> List[SearchResult]:
        """
        Execute a single search query using Gemini with Google Search grounding.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of SearchResult objects
        """
        # Check cache first
        cache_key = f"{query}_{max_results}"
        if cache_key in self.search_cache:
            logger.debug(f"Using cached results for query: '{query}'")
            return self.search_cache[cache_key]
        
        try:
            # Create search grounding tool
            grounding_tool = types.Tool(google_search=types.GoogleSearch())
            config = types.GenerateContentConfig(tools=[grounding_tool])
            
            # Construct search prompt
            search_prompt = self._construct_search_prompt(query, max_results)
            
            # Execute search via Gemini API
            logger.debug(f"Executing search for: '{query}'")
            response = self.client.models.generate_content(
                model=self.config.gemini_model,
                contents=search_prompt,
                config=config
            )
            
            # Parse search results
            results = self._parse_search_response(query, response)
            
            # Cache results
            self.search_cache[cache_key] = results
            
            # Track token usage if tokenizer available
            if self.tokenizer and hasattr(response, 'usage_metadata'):
                self.tokenizer.log_token_usage(
                    operation=f"search_{query[:50]}",
                    input_tokens=getattr(response.usage_metadata, 'prompt_token_count', 0),
                    output_tokens=getattr(response.usage_metadata, 'candidates_token_count', 0),
                    cost=self.cost_per_search
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []
    
    def _construct_search_prompt(self, query: str, max_results: int) -> str:
        """Construct an optimized search prompt for Gemini."""
        prompt = f"""Please search for information about: {query}

I need you to find {max_results} relevant web sources. For each source you find, please provide:
1. The URL/link
2. The title of the page/article
3. A brief summary or snippet of the content
4. Why this source is relevant to the query

Focus on finding:
- Recent and authoritative sources
- Diverse perspectives and source types
- High-quality content (research papers, reputable news, official documentation)

Please prioritize sources published in the last 2 years when possible."""

        return prompt
    
    def _parse_search_response(self, query: str, response) -> List[SearchResult]:
        """
        Parse Gemini search response to extract structured results.
        
        Args:
            query: Original search query
            response: Gemini API response with grounding metadata
            
        Returns:
            List of SearchResult objects
        """
        results = []
        
        try:
            # Check if response has grounding metadata
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                
                # Extract grounding metadata if available
                if hasattr(candidate, 'grounding_metadata'):
                    grounding = candidate.grounding_metadata
                    
                    # Process grounding supports (search results)
                    if hasattr(grounding, 'grounding_supports'):
                        for support in grounding.grounding_supports:
                            # Extract URL and metadata
                            if hasattr(support, 'web_search') and hasattr(support.web_search, 'uri'):
                                url = support.web_search.uri
                                title = getattr(support.web_search, 'title', 'Untitled')
                                
                                # Create search result
                                result = SearchResult(
                                    query=query,
                                    url=url,
                                    title=title,
                                    snippet="", # Will be filled by content extraction
                                    relevance_score=0.8,  # Default high relevance from grounding
                                    source_type=self._classify_source_type(url, title)
                                )
                                results.append(result)
            
            # If no grounding metadata, try to extract URLs from response text
            if not results and hasattr(response, 'text'):
                text_results = self._extract_urls_from_text(query, response.text)
                results.extend(text_results)
            
            # Score and rank results
            results = self._score_search_results(results, query)
            
        except Exception as e:
            logger.error(f"Error parsing search response: {e}")
        
        return results[:self.config.search_results_per_query]
    
    def _extract_urls_from_text(self, query: str, text: str) -> List[SearchResult]:
        """Extract URLs and context from response text as fallback."""
        results = []
        
        # Simple URL extraction pattern
        import re
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        
        for url in urls:
            # Extract surrounding context as title/snippet
            url_pos = text.find(url)
            context_start = max(0, url_pos - 100)
            context_end = min(len(text), url_pos + len(url) + 100)
            context = text[context_start:context_end].strip()
            
            # Extract potential title (text before URL)
            lines = context.split('\n')
            title = "Untitled"
            for line in lines:
                if url not in line and len(line.strip()) > 10:
                    title = line.strip()
                    break
            
            result = SearchResult(
                query=query,
                url=url,
                title=title[:200],  # Limit title length
                snippet=context[:500],  # Limit snippet length
                relevance_score=0.6,  # Lower relevance for text extraction
                source_type=self._classify_source_type(url, title)
            )
            results.append(result)
        
        return results
    
    def _classify_source_type(self, url: str, title: str) -> str:
        """Classify the type of source based on URL and title patterns."""
        url_lower = url.lower()
        title_lower = title.lower()
        
        # Academic/Research sources
        if any(domain in url_lower for domain in [
            'arxiv.org', 'scholar.google', 'researchgate', 'ieee.org', 
            'acm.org', 'nature.com', 'science.org', 'pubmed'
        ]):
            return 'research_paper'
        
        # Documentation and official sources
        if any(pattern in url_lower for pattern in [
            'docs.', '/documentation/', '/api/', 'developer.', 'github.com',
            '.gov', 'wikipedia.org'
        ]):
            return 'documentation'
        
        # News and media
        if any(domain in url_lower for domain in [
            'reuters.com', 'bloomberg.com', 'techcrunch.com', 'wired.com',
            'cnn.com', 'bbc.com', 'forbes.com', 'wsj.com'
        ]):
            return 'news'
        
        # Blogs and opinion
        if any(pattern in url_lower for pattern in [
            'medium.com', 'substack.com', 'blog', 'towards'
        ]):
            return 'blog'
        
        # Default classification
        return 'article'
    
    def _score_search_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """
        Score and rank search results based on relevance factors.
        
        Args:
            results: List of SearchResult objects
            query: Original search query
            
        Returns:
            Scored and sorted list of SearchResult objects
        """
        query_words = set(query.lower().split())
        
        for result in results:
            score = result.relevance_score  # Start with base score
            
            # Title relevance
            title_words = set(result.title.lower().split())
            title_overlap = len(query_words.intersection(title_words))
            score += title_overlap * 0.3
            
            # URL relevance
            if any(word in result.url.lower() for word in query_words):
                score += 0.2
            
            # Source type bonus
            source_bonuses = {
                'research_paper': 0.4,
                'documentation': 0.3,
                'news': 0.2,
                'article': 0.1,
                'blog': 0.05
            }
            score += source_bonuses.get(result.source_type, 0)
            
            # Recency bonus (if we can detect dates)
            current_year = datetime.now().year
            if str(current_year) in result.title or str(current_year) in result.url:
                score += 0.1
            elif str(current_year - 1) in result.title or str(current_year - 1) in result.url:
                score += 0.05
            
            result.relevance_score = min(score, 2.0)  # Cap at 2.0
        
        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results
    
    def _update_search_costs(self, search_count: int) -> None:
        """Update search cost tracking."""
        cost = search_count * self.cost_per_search
        self.search_costs += cost
        self.search_count += search_count
        
        if self.config.enable_cost_tracking:
            logger.info(f"Search cost update: {search_count} searches = ${cost:.3f} "
                       f"(Total: ${self.search_costs:.3f})")
        
        # Check cost alerts
        if (self.search_costs > self.config.cost_alert_threshold and 
            self.config.enable_cost_tracking):
            logger.warning(f"Search costs exceeded alert threshold: ${self.search_costs:.2f}")
    
    def get_search_summary(self) -> Dict[str, Any]:
        """Get comprehensive search operation summary."""
        return {
            'total_searches': self.search_count,
            'total_cost': self.search_costs,
            'average_cost_per_search': (
                self.search_costs / self.search_count if self.search_count > 0 else 0
            ),
            'cache_size': len(self.search_cache),
            'api_configured': self.client is not None,
            'search_cache_keys': list(self.search_cache.keys())[:5]  # First 5 for debugging
        }
    
    def clear_search_cache(self) -> int:
        """Clear search cache and return number of entries cleared."""
        count = len(self.search_cache)
        self.search_cache.clear()
        logger.info(f"Search cache cleared: {count} entries removed")
        return count
    
    async def search_with_strategy(self, search_strategy: Dict[str, Any]) -> List[SearchResult]:
        """
        Execute searches based on a comprehensive search strategy.
        
        Args:
            search_strategy: Strategy dict from KeywordExtractor.generate_search_strategy()
            
        Returns:
            List of SearchResult objects ranked by relevance
        """
        all_results = []
        
        # Execute primary phase searches first
        primary_queries = search_strategy['search_phases']['primary']
        if primary_queries:
            logger.info(f"Executing primary search phase: {len(primary_queries)} queries")
            primary_results = await self.search_multiple_queries(primary_queries)
            all_results.extend(primary_results)
        
        # Check if we have enough results or should continue
        remaining_budget = self.config.max_searches_per_task - len(primary_queries)
        secondary_queries = search_strategy['search_phases']['secondary']
        
        if remaining_budget > 0 and secondary_queries:
            # Execute secondary searches with remaining budget
            queries_to_run = secondary_queries[:remaining_budget]
            logger.info(f"Executing secondary search phase: {len(queries_to_run)} queries")
            secondary_results = await self.search_multiple_queries(queries_to_run)
            all_results.extend(secondary_results)
        
        # Remove duplicates based on URL
        unique_results = []
        seen_urls = set()
        for result in all_results:
            if result.url and result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        # Final ranking
        unique_results = self._score_search_results(
            unique_results, 
            search_strategy['original_query']
        )
        
        logger.info(f"Search strategy completed: {len(unique_results)} unique results")
        return unique_results


@dataclass
class CrawledContent:
    """Represents crawled web content with metadata."""
    url: str
    title: str = ""
    content: str = ""
    content_length: int = 0
    token_count: int = 0
    success: bool = False
    error_message: str = ""
    crawl_time: float = 0.0
    content_type: str = "html"
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'url': self.url,
            'title': self.title,
            'content': self.content,
            'content_length': self.content_length,
            'token_count': self.token_count,
            'success': self.success,
            'error_message': self.error_message,
            'crawl_time': self.crawl_time,
            'content_type': self.content_type,
            'timestamp': self.timestamp.isoformat()
        }


class WebContentExtractor:
    """
    Web content extraction using Crawl4AI for intelligent web scraping.
    
    This class handles:
    1. Parallel URL crawling with Crawl4AI
    2. Content cleaning and filtering
    3. Token-aware content optimization
    4. Error handling and timeouts
    5. Content quality assessment
    """
    
    def __init__(self, config: ResearchConfig, tokenizer: Optional['TokenCounter'] = None):
        """Initialize the WebContentExtractor."""
        self.config = config
        self.tokenizer = tokenizer
        self.crawler = None
        self.total_crawl_time = 0.0
        self.successful_crawls = 0
        self.failed_crawls = 0
        
        # Initialize async web crawler
        if AsyncWebCrawler:
            try:
                # Configure crawler settings
                crawler_config = {
                    'headless': True,
                    'verbose': self.config.debug_mode,
                }
                self.crawler_initialized = True
                logger.info("WebContentExtractor initialized with Crawl4AI")
            except Exception as e:
                logger.error(f"Failed to initialize Crawl4AI: {e}")
                self.crawler_initialized = False
        else:
            logger.error("Crawl4AI library not installed")
            self.crawler_initialized = False
    
    async def extract_content_from_urls(self, search_results: List[SearchResult]) -> List[CrawledContent]:
        """
        Extract content from multiple URLs in parallel.
        
        Args:
            search_results: List of SearchResult objects with URLs to crawl
            
        Returns:
            List of CrawledContent objects
        """
        if not self.crawler_initialized:
            logger.error("Crawl4AI not initialized - cannot extract content")
            return []
        
        # Filter out URLs and check for paywall blacklist
        filtered_results = []
        skipped_paywalls = []
        
        for result in search_results:
            if not result.url:
                continue
                
            # Check if URL is in paywall blacklist
            if self._is_url_in_paywall_blacklist(result.url):
                skipped_paywalls.append(result.url)
                # Create failed crawl result for paywall
                paywall_content = CrawledContent(url=result.url, title=result.title)
                paywall_content.success = False
                paywall_content.error_message = "URL in paywall blacklist - skipped"
                filtered_results.append(paywall_content)
            else:
                filtered_results.append(result)
        
        if skipped_paywalls:
            logger.info(f"Skipped {len(skipped_paywalls)} paywall URLs: {skipped_paywalls}")
        
        # Get URLs that will actually be crawled
        urls_to_crawl = [result.url for result in filtered_results if isinstance(result, SearchResult)]
        
        if not urls_to_crawl:
            logger.warning("No valid URLs found after paywall filtering")
            return [result for result in filtered_results if isinstance(result, CrawledContent)]
        
        logger.info(f"Extracting content from {len(urls_to_crawl)} URLs (skipped {len(skipped_paywalls)} paywall sites)")
        
        # Limit concurrent crawls based on configuration
        max_concurrent = min(len(urls_to_crawl), self.config.max_concurrent_crawls)
        
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Execute crawls with timeout
        crawl_timeout = self.config.max_crawl_timeout_minutes * 60  # Convert to seconds
        
        try:
            # Create crawl tasks for non-paywall URLs only
            valid_search_results = [result for result in filtered_results if isinstance(result, SearchResult)]
            tasks = [
                self._crawl_single_url_with_semaphore(semaphore, result.url, result.title, result.source_type)
                for result in valid_search_results
            ]
            
            # Execute with global timeout
            crawled_content = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=crawl_timeout
            )
            
        except asyncio.TimeoutError:
            logger.warning(f"Crawling timed out after {crawl_timeout} seconds")
            # Return whatever we have so far
            crawled_content = []
        
        # Combine paywall-skipped content with crawled content
        paywall_content = [result for result in filtered_results if isinstance(result, CrawledContent)]
        
        # Filter out exceptions and process results
        valid_content = list(paywall_content)  # Start with paywall-skipped content
        
        for result in crawled_content:
            if isinstance(result, CrawledContent):
                valid_content.append(result)
                if result.success:
                    self.successful_crawls += 1
                else:
                    self.failed_crawls += 1
            elif isinstance(result, Exception):
                logger.error(f"Crawl error: {result}")
                self.failed_crawls += 1
        
        # Optimize content for token limits
        if self.tokenizer:
            valid_content = self._optimize_content_for_tokens(valid_content)
        
        logger.info(f"Content extraction completed: {self.successful_crawls} successful, {self.failed_crawls} failed")
        
        return valid_content
    
    async def _crawl_single_url_with_semaphore(self, semaphore: asyncio.Semaphore, 
                                             url: str, title: str = "", source_type: str = "") -> CrawledContent:
        """Crawl a single URL with semaphore for concurrency control."""
        async with semaphore:
            return await self._crawl_single_url(url, title, source_type)
    
    async def _crawl_single_url(self, url: str, title: str = "", source_type: str = "") -> CrawledContent:
        """
        Crawl a single URL and extract content.
        
        Args:
            url: URL to crawl
            title: Optional title from search results
            source_type: Type of source (research_paper, news, etc.)
            
        Returns:
            CrawledContent object
        """
        start_time = datetime.now()
        content_obj = CrawledContent(url=url, title=title)
        
        try:
            # Import Crawl4AI components for clean markdown generation
            try:
                from crawl4ai import CrawlerRunConfig, DefaultMarkdownGenerator, PruningContentFilter
                use_clean_markdown = True
            except ImportError:
                use_clean_markdown = False
                
            # Initialize crawler for this session
            async with AsyncWebCrawler(
                headless=True,
                verbose=False  # Reduce noise in logs
            ) as crawler:
                
                # Configure crawling parameters with clean markdown generation
                crawl_config = {
                    'bypass_cache': False,
                    'css_selector': self._get_content_selector(source_type),
                    'word_count_threshold': self.config.min_content_length,
                    'extraction_strategy': "NoExtractionStrategy",  # Get raw content first
                    'chunking_strategy': "RegexChunking",
                    'excluded_tags': ['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'input'],
                    'remove_overlay_elements': True,
                }
                
                # Add clean markdown configuration if available
                if use_clean_markdown:
                    crawl_config['crawler_config'] = CrawlerRunConfig(
                        markdown_generator=DefaultMarkdownGenerator(
                            content_filter=PruningContentFilter(
                                threshold=0.48,  # Remove low-relevance content
                                threshold_type="fixed",
                                min_word_threshold=10  # Minimum words to keep content
                            ),
                            options={
                                "ignore_links": True,  # Remove hyperlinks for cleaner text
                                "ignore_images": False,  # Keep image references
                                "body_width": 120  # Format text width
                            }
                        )
                    )
                
                # Execute crawl with individual timeout
                crawl_result = await asyncio.wait_for(
                    crawler.arun(url=url, **crawl_config),
                    timeout=self.config.crawl_timeout_per_url
                )
                
                # Process crawl result
                if crawl_result.success:
                    content_obj.success = True
                    
                    # Use clean markdown format if available, otherwise fallback to regular markdown
                    if use_clean_markdown and hasattr(crawl_result, 'markdown') and hasattr(crawl_result.markdown, 'fit_markdown'):
                        # Use filtered/cleaned markdown with links removed
                        raw_content = crawl_result.markdown.fit_markdown or crawl_result.markdown.raw_markdown
                        logger.debug(f"Using fit_markdown for {url}")
                    else:
                        # Fallback to regular markdown or cleaned HTML
                        raw_content = crawl_result.markdown or crawl_result.cleaned_html
                        logger.debug(f"Using regular markdown for {url}")
                    
                    content_obj.content = self._clean_content(raw_content)
                    content_obj.content_length = len(content_obj.content)
                    content_obj.title = content_obj.title or crawl_result.title or self._extract_title_from_url(url)
                    
                    # Check for paywall or verification pages
                    if self._is_paywall_or_blocked(content_obj.content):
                        content_obj.success = False
                        content_obj.error_message = "Content blocked by paywall or verification page"
                        logger.warning(f"Paywall/verification detected for {url}: {content_obj.content[:100]}...")
                        return content_obj
                    
                    # Count tokens if tokenizer available
                    if self.tokenizer:
                        content_obj.token_count = self.tokenizer.count_tokens(content_obj.content)
                    else:
                        # Rough estimate: 4 chars per token
                        content_obj.token_count = content_obj.content_length // 4
                    
                    logger.debug(f"Successfully crawled {url}: {content_obj.content_length} chars, "
                               f"{content_obj.token_count} tokens")
                
                else:
                    content_obj.success = False
                    content_obj.error_message = f"Crawl failed: {crawl_result.status_code}"
                    logger.warning(f"Failed to crawl {url}: {content_obj.error_message}")
        
        except asyncio.TimeoutError:
            content_obj.success = False
            content_obj.error_message = f"Timeout after {self.config.crawl_timeout_per_url}s"
            logger.warning(f"Crawl timeout for {url}")
            
        except Exception as e:
            content_obj.success = False
            content_obj.error_message = str(e)
            logger.error(f"Crawl error for {url}: {e}")
        
        # Calculate crawl time
        end_time = datetime.now()
        content_obj.crawl_time = (end_time - start_time).total_seconds()
        self.total_crawl_time += content_obj.crawl_time
        
        return content_obj
    
    def _get_content_selector(self, source_type: str) -> str:
        """Get appropriate CSS selector based on source type."""
        selectors = {
            'research_paper': 'article, .paper-content, .abstract, .content, main',
            'news': 'article, .article-content, .story-body, .content, main',
            'documentation': '.mw-parser-output p, .mw-content-text > p, .bodytext, article',
            'blog': 'article, .post-content, .entry-content, .content, main',
            'article': 'article, .article-content, .content, main'
        }
        return selectors.get(source_type, 'article, .content, main')
    
    def _clean_content(self, raw_content: str) -> str:
        """Clean and normalize extracted content, removing HTML links and noise."""
        if not raw_content:
            return ""
        
        import re
        content = raw_content
        
        # Remove markdown links but keep the text: [text](url) -> text
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
        
        # Remove HTML-style links: <a href="...">text</a> -> text
        content = re.sub(r'<a[^>]*>([^<]+)</a>', r'\1', content)
        
        # Remove Wikipedia-style citations: [[text]] -> text
        content = re.sub(r'\[\[([^\]]+)\]\]', r'\1', content)
        
        # Remove URL patterns that might remain
        content = re.sub(r'https?://[^\s]+', '', content)
        
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        content = re.sub(r' +', ' ', content)
        content = re.sub(r'\t+', ' ', content)
        
        # Remove Wikipedia-specific navigation and content structure
        # Remove table of contents section
        content = re.sub(r'## Contents\s*move to sidebar hide.*?(?=##|\n\n[A-Z])', '', content, flags=re.DOTALL)
        
        # Remove navigation sections at the beginning
        nav_sections = [
            r'^Jump to content.*?(?=\n\n[A-Z])',
            r'^Main \s*Main \s*move to sidebar hide.*?(?=\n\n[A-Z])',
            r'^\s*\* Main page.*?(?=\n\n[A-Z])',
            r'^Contribute \s*\* Help.*?(?=\n\n[A-Z])',
            r'^Personal tools.*?(?=\n\n[A-Z])',
            r'^Pages for logged out editors.*?(?=\n\n[A-Z])',
            r'^Appearance.*?(?=\n\n[A-Z])',
        ]
        
        for pattern in nav_sections:
            content = re.sub(pattern, '', content, flags=re.MULTILINE | re.DOTALL)
        
        # Remove common navigation and boilerplate text
        noise_patterns = [
            r'Cookie policy.*?(?=\n|$)',
            r'Privacy policy.*?(?=\n|$)', 
            r'Terms of service.*?(?=\n|$)',
            r'Subscribe to.*?(?=\n|$)',
            r'Share this.*?(?=\n|$)',
            r'Follow us.*?(?=\n|$)',
            r'Advertisement.*?(?=\n|$)',
            r'Sign up for.*?(?=\n|$)',
            r'Log in.*?(?=\n|$)',
            r'Skip to.*?(?=\n|$)',
            r'Menu.*?(?=\n|$)',
            r'Navigation.*?(?=\n|$)',
            r'move to sidebar hide.*?(?=\n|$)',
            r'Toggle.*?subsection.*?(?=\n|$)',
            r'learn more.*?(?=\n|$)',
            r'Search\s*Search.*?(?=\n|$)',
            r'Donate.*?Create account.*?(?=\n|$)',
        ]
        
        for pattern in noise_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        # Trim to max length if specified
        if (self.config.max_content_length_per_url and 
            len(content) > self.config.max_content_length_per_url):
            content = content[:self.config.max_content_length_per_url] + "...\n\n[Content truncated due to length limit]"
        
        return content.strip()
    
    def _is_url_in_paywall_blacklist(self, url: str) -> bool:
        """Check if URL matches any paywall blacklist patterns."""
        if not self.config.paywall_url_blacklist:
            return False
        
        url_lower = url.lower()
        
        for blacklist_pattern in self.config.paywall_url_blacklist:
            pattern_lower = blacklist_pattern.lower()
            
            # Check if URL contains the blacklist pattern
            if pattern_lower in url_lower:
                return True
        
        return False
    
    def _is_paywall_or_blocked(self, content: str) -> bool:
        """
        Detect if content is blocked by paywall or verification page.
        
        Args:
            content: Extracted content to check
            
        Returns:
            True if content appears to be paywall/verification, False otherwise
        """
        if not content or len(content.strip()) < self.config.min_legitimate_content_length:
            return True
        
        # Common paywall/verification indicators
        paywall_indicators = [
            # Chinese verification/error pages
            "环境异常", "完成验证", "去验证", "当前环境异常",
            "访问异常", "验证失败", "请稍后重试", "系统维护",
            
            # English paywall indicators
            "subscribe to continue", "sign in to read", "create account",
            "paywall", "premium content", "membership required",
            "register to continue", "login required", "access denied",
            "verification required", "captcha", "robot verification",
            "rate limit", "too many requests", "blocked",
            
            # Generic error messages
            "403 forbidden", "404 not found", "500 error",
            "access restricted", "content not available",
            "please enable javascript", "cookies required",
            
            # Social media login walls
            "continue with facebook", "continue with google",
            "create free account", "join to see more",
            
            # News site paywalls
            "this article is for subscribers", "become a subscriber",
            "free trial", "unlock full access", "read premium",
        ]
        
        content_lower = content.lower()
        
        # Check for paywall indicators
        paywall_count = sum(1 for indicator in paywall_indicators if indicator in content_lower)
        
        # If multiple paywall indicators found, likely blocked
        if paywall_count >= 2:
            return True
        
        # Check content structure - legitimate articles have paragraphs
        paragraphs = [p for p in content.split('\n\n') if len(p.strip()) > 30]
        if len(paragraphs) < 2:
            return True
        
        # Check for repetitive content (often indicates navigation/error pages)
        words = content_lower.split()
        if len(words) < 50:  # Very short content
            return True
        
        # Check word variety (real articles have diverse vocabulary)
        unique_words = len(set(words))
        word_variety_ratio = unique_words / len(words) if words else 0
        
        if word_variety_ratio < 0.3:  # Less than 30% unique words suggests repetitive content
            return True
        
        return False
    
    def _extract_title_from_url(self, url: str) -> str:
        """Extract a reasonable title from URL if no title available."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            
            # Try to get meaningful part from path
            path_parts = [part for part in parsed.path.split('/') if part and part != 'index.html']
            if path_parts:
                title = path_parts[-1].replace('-', ' ').replace('_', ' ')
                return title.title()
            
            return parsed.netloc
        except:
            return "Untitled"
    
    def _optimize_content_for_tokens(self, content_list: List[CrawledContent]) -> List[CrawledContent]:
        """Optimize content list to fit within token limits."""
        if not self.tokenizer:
            return content_list
        
        # Sort by success and content quality (length, source type relevance)
        def content_quality_score(content: CrawledContent) -> float:
            score = 0.0
            if content.success:
                score += 10.0
            
            # Content length factor (prefer substantial content)
            if 1000 <= content.content_length <= 10000:
                score += 5.0
            elif content.content_length > 500:
                score += 3.0
            elif content.content_length > 100:
                score += 1.0
            
            return score
        
        # Sort by quality score (highest first)
        content_list.sort(key=content_quality_score, reverse=True)
        
        # Use tokenizer to optimize for context window
        content_texts = [c.content for c in content_list if c.success]
        optimized_texts, optimization_stats = self.tokenizer.optimize_content_for_context(
            content_texts, 
            max_tokens=self.config.max_total_tokens
        )
        
        if self.config.debug_mode:
            logger.info(f"Content optimization: {optimization_stats}")
        
        # Update content list with optimized content
        optimized_content = []
        optimized_index = 0
        
        for content in content_list:
            if content.success and optimized_index < len(optimized_texts):
                # Update with optimized content
                content.content = optimized_texts[optimized_index]
                content.content_length = len(content.content)
                if self.tokenizer:
                    content.token_count = self.tokenizer.count_tokens(content.content)
                optimized_content.append(content)
                optimized_index += 1
            elif not content.success:
                # Keep failed attempts for debugging
                optimized_content.append(content)
        
        return optimized_content
    
    def get_crawl_summary(self) -> Dict[str, Any]:
        """Get comprehensive crawling operation summary."""
        return {
            'total_crawl_time': self.total_crawl_time,
            'successful_crawls': self.successful_crawls,
            'failed_crawls': self.failed_crawls,
            'total_attempts': self.successful_crawls + self.failed_crawls,
            'success_rate': (
                self.successful_crawls / (self.successful_crawls + self.failed_crawls)
                if (self.successful_crawls + self.failed_crawls) > 0 else 0
            ),
            'average_crawl_time': (
                self.total_crawl_time / (self.successful_crawls + self.failed_crawls)
                if (self.successful_crawls + self.failed_crawls) > 0 else 0
            ),
            'crawler_initialized': self.crawler_initialized
        }
    
    async def extract_content_with_strategy(self, search_results: List[SearchResult]) -> Dict[str, Any]:
        """
        Extract content using intelligent strategy based on search results.
        
        Returns comprehensive extraction results with metadata.
        """
        if not search_results:
            return {
                'crawled_content': [],
                'extraction_summary': self.get_crawl_summary(),
                'total_content_length': 0,
                'total_token_count': 0
            }
        
        # Prioritize results by relevance score
        sorted_results = sorted(search_results, key=lambda x: x.relevance_score, reverse=True)
        
        # Take top results within our crawl limit
        max_urls = min(len(sorted_results), self.config.max_concurrent_crawls * 2)  # Allow more URLs
        priority_results = sorted_results[:max_urls]
        
        logger.info(f"Extracting content from {len(priority_results)} prioritized URLs")
        
        # Extract content
        crawled_content = await self.extract_content_from_urls(priority_results)
        
        # Calculate totals
        total_content_length = sum(c.content_length for c in crawled_content if c.success)
        total_token_count = sum(c.token_count for c in crawled_content if c.success)
        
        return {
            'crawled_content': crawled_content,
            'extraction_summary': self.get_crawl_summary(),
            'total_content_length': total_content_length,
            'total_token_count': total_token_count,
            'successful_urls': [c.url for c in crawled_content if c.success],
            'failed_urls': [c.url for c in crawled_content if not c.success]
        }


class TokenCounter:
    """
    Advanced tokenizer for accurate token counting and management.
    
    Supports multiple methods for token estimation:
    1. tiktoken (most accurate for OpenAI-compatible models)
    2. Character-based estimation (fallback method)
    3. Word-based estimation (rough approximation)
    """
    
    def __init__(self, config: ResearchConfig):
        """Initialize the TokenCounter with configuration."""
        self.config = config
        self.encoding = None
        
        # Try to initialize tiktoken encoder
        if tiktoken:
            try:
                # Use cl100k_base encoding (GPT-4 compatible, close to Gemini)
                self.encoding = tiktoken.get_encoding("cl100k_base")
                logger.info("TokenCounter initialized with tiktoken encoding")
            except Exception as e:
                logger.warning(f"Failed to initialize tiktoken: {e}")
                self.encoding = None
        
        # Token usage tracking
        self.total_usage = TokenUsage()
        self.session_usage = []
        
        logger.info(f"TokenCounter initialized - Input limit: {config.gemini_input_context_limit:,}, "
                   f"Output limit: {config.gemini_output_context_limit:,}")
    
    def count_tokens(self, text: str, method: str = "auto") -> int:
        """
        Count tokens in text using specified method.
        
        Args:
            text: Input text to count tokens for
            method: Counting method ("auto", "tiktoken", "character", "word")
            
        Returns:
            Estimated number of tokens
        """
        if not text or not isinstance(text, str):
            return 0
        
        if method == "auto":
            # Use best available method
            if self.encoding:
                return self._count_tokens_tiktoken(text)
            else:
                return self._count_tokens_character(text)
        elif method == "tiktoken" and self.encoding:
            return self._count_tokens_tiktoken(text)
        elif method == "character":
            return self._count_tokens_character(text)
        elif method == "word":
            return self._count_tokens_word(text)
        else:
            # Fallback to character-based counting
            return self._count_tokens_character(text)
    
    def _count_tokens_tiktoken(self, text: str) -> int:
        """Count tokens using tiktoken (most accurate)."""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.warning(f"tiktoken encoding failed: {e}, falling back to character method")
            return self._count_tokens_character(text)
    
    def _count_tokens_character(self, text: str) -> int:
        """
        Estimate tokens using character count.
        
        Rule of thumb: ~4 characters per token for English text
        More accurate for mixed content including code/data
        """
        # More sophisticated character-based estimation
        char_count = len(text)
        
        # Adjust ratio based on content type
        if self._is_code_heavy(text):
            # Code typically has fewer chars per token
            ratio = 3.5
        elif self._has_special_characters(text):
            # Special characters and punctuation
            ratio = 3.8
        else:
            # Regular text
            ratio = 4.0
        
        return int(char_count / ratio)
    
    def _count_tokens_word(self, text: str) -> int:
        """
        Rough estimation using word count.
        
        Rule of thumb: ~0.75 tokens per word
        Less accurate but very fast
        """
        words = len(text.split())
        return int(words * 0.75)
    
    def _is_code_heavy(self, text: str) -> bool:
        """Detect if text contains significant code content."""
        code_indicators = ['{', '}', '()', '=>', 'function', 'class', 'def ', 'import ', '#include']
        code_count = sum(1 for indicator in code_indicators if indicator in text)
        return code_count >= 3
    
    def _has_special_characters(self, text: str) -> bool:
        """Detect if text has many special characters."""
        special_chars = re.findall(r'[^\w\s]', text)
        return len(special_chars) / len(text) > 0.1 if text else False
    
    def estimate_tokens_for_content_list(self, content_list: List[str]) -> Tuple[int, List[int]]:
        """
        Estimate tokens for a list of content pieces.
        
        Returns:
            Tuple of (total_tokens, individual_token_counts)
        """
        individual_counts = []
        total_tokens = 0
        
        for content in content_list:
            count = self.count_tokens(content)
            individual_counts.append(count)
            total_tokens += count
        
        return total_tokens, individual_counts
    
    def can_fit_in_context(self, current_tokens: int, additional_text: str = "", 
                          reserve_output_tokens: bool = True) -> Tuple[bool, Dict[str, int]]:
        """
        Check if additional content can fit in context window.
        
        Args:
            current_tokens: Current token count
            additional_text: Text to add
            reserve_output_tokens: Whether to reserve space for output
            
        Returns:
            Tuple of (can_fit, token_breakdown)
        """
        additional_tokens = self.count_tokens(additional_text) if additional_text else 0
        reserved_tokens = self.config.reserved_tokens_for_report if reserve_output_tokens else 0
        safety_buffer = self.config.token_safety_buffer
        
        total_needed = current_tokens + additional_tokens + reserved_tokens + safety_buffer
        available_tokens = self.config.gemini_input_context_limit
        
        can_fit = total_needed <= available_tokens
        
        breakdown = {
            'current_tokens': current_tokens,
            'additional_tokens': additional_tokens,
            'reserved_for_output': reserved_tokens,
            'safety_buffer': safety_buffer,
            'total_needed': total_needed,
            'available_tokens': available_tokens,
            'remaining_tokens': available_tokens - total_needed,
            'can_fit': can_fit
        }
        
        return can_fit, breakdown
    
    def optimize_content_for_context(self, content_list: List[str], 
                                   max_tokens: Optional[int] = None) -> Tuple[List[str], Dict[str, Any]]:
        """
        Optimize content list to fit within token limits.
        
        Args:
            content_list: List of content pieces
            max_tokens: Maximum tokens allowed (uses config default if None)
            
        Returns:
            Tuple of (optimized_content_list, optimization_stats)
        """
        if max_tokens is None:
            max_tokens = self.config.max_total_tokens
        
        # Calculate tokens for each piece
        total_tokens, individual_counts = self.estimate_tokens_for_content_list(content_list)
        
        stats = {
            'original_count': len(content_list),
            'original_tokens': total_tokens,
            'target_max_tokens': max_tokens,
            'pieces_removed': 0,
            'pieces_truncated': 0,
            'final_tokens': 0,
            'final_count': 0
        }
        
        if total_tokens <= max_tokens:
            # Content already fits
            stats['final_tokens'] = total_tokens
            stats['final_count'] = len(content_list)
            return content_list, stats
        
        # Content doesn't fit - optimize
        optimized_content = []
        current_tokens = 0
        
        # Sort by length (keep shorter pieces first - they're likely more focused)
        content_with_tokens = list(zip(content_list, individual_counts))
        content_with_tokens.sort(key=lambda x: x[1])  # Sort by token count
        
        for content, token_count in content_with_tokens:
            if current_tokens + token_count <= max_tokens:
                # Piece fits entirely
                optimized_content.append(content)
                current_tokens += token_count
            else:
                # Try to truncate the piece to fit remaining space
                remaining_space = max_tokens - current_tokens
                if remaining_space > 1000:  # Only truncate if we have reasonable space
                    # Truncate to fit
                    ratio = remaining_space / token_count
                    truncate_length = int(len(content) * ratio * 0.9)  # 90% safety margin
                    
                    if truncate_length > 500:  # Only if truncated piece is still meaningful
                        truncated_content = content[:truncate_length] + "..."
                        truncated_tokens = self.count_tokens(truncated_content)
                        
                        if current_tokens + truncated_tokens <= max_tokens:
                            optimized_content.append(truncated_content)
                            current_tokens += truncated_tokens
                            stats['pieces_truncated'] += 1
                            break
                
                # Piece can't fit - skip it
                stats['pieces_removed'] += 1
        
        stats['final_tokens'] = current_tokens
        stats['final_count'] = len(optimized_content)
        
        if self.config.debug_mode:
            logger.info(f"Content optimization: {stats}")
        
        return optimized_content, stats
    
    def log_token_usage(self, operation: str, input_tokens: int = 0, 
                       output_tokens: int = 0, cost: float = 0.0) -> None:
        """Log token usage for an operation."""
        usage = TokenUsage()
        usage.add_usage(input_tokens, output_tokens)
        usage.estimated_cost = cost
        
        self.total_usage.add_usage(input_tokens, output_tokens)
        self.total_usage.estimated_cost += cost
        
        self.session_usage.append({
            'operation': operation,
            'timestamp': datetime.now().isoformat(),
            'usage': usage.to_dict()
        })
        
        if self.config.debug_mode:
            logger.info(f"Token usage - {operation}: {input_tokens:,} in, {output_tokens:,} out, ${cost:.4f}")
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive token usage summary."""
        return {
            'total_usage': self.total_usage.to_dict(),
            'session_operations': len(self.session_usage),
            'average_tokens_per_operation': (
                self.total_usage.total_tokens / len(self.session_usage) 
                if self.session_usage else 0
            ),
            'context_utilization': (
                self.total_usage.input_tokens / self.config.gemini_input_context_limit
            ) * 100,
            'estimated_total_cost': self.total_usage.estimated_cost,
            'session_history': self.session_usage[-10:]  # Last 10 operations
        }
    
    def reset_usage_tracking(self) -> Dict[str, Any]:
        """Reset usage tracking and return final summary."""
        summary = self.get_usage_summary()
        self.total_usage = TokenUsage()
        self.session_usage = []
        return summary


# Test the tokenizer functionality
def test_tokenizer():
    """Test function for the TokenCounter class."""
    print("Testing TokenCounter...")
    
    config = ResearchConfig()
    tokenizer = TokenCounter(config)
    
    # Test different types of content
    test_texts = [
        "Hello world, this is a simple test.",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 50,
        "{'key': 'value', 'numbers': [1, 2, 3, 4, 5], 'nested': {'a': 'b'}}"
    ]
    
    for i, text in enumerate(test_texts, 1):
        tokens = tokenizer.count_tokens(text)
        print(f"Test {i}: {tokens:,} tokens ({len(text)} chars)")
        
        # Test context fitting
        can_fit, breakdown = tokenizer.can_fit_in_context(tokens * 100, text)
        print(f"  Can fit 100x this content: {can_fit}")
        print(f"  Breakdown: {breakdown['total_needed']:,}/{breakdown['available_tokens']:,} tokens")
    
    # Test content optimization
    long_content_list = [text * 1000 for text in test_texts]  # Make content very long
    optimized, stats = tokenizer.optimize_content_for_context(long_content_list, max_tokens=50000)
    
    print(f"\nContent optimization test:")
    print(f"  Original: {stats['original_count']} pieces, {stats['original_tokens']:,} tokens")
    print(f"  Optimized: {stats['final_count']} pieces, {stats['final_tokens']:,} tokens")
    print(f"  Removed: {stats['pieces_removed']}, Truncated: {stats['pieces_truncated']}")
    
    print("\nTokenizer test completed!")


class AIContentPolisher:
    """
    AI-powered content polishing and comprehensive report generation.
    
    This class uses Gemini Pro 2.5 to:
    1. Polish and clean extracted content
    2. Generate comprehensive research reports
    3. Create blog-ready markdown content
    """
    
    def __init__(self, config: ResearchConfig, tokenizer: Optional['TokenCounter'] = None):
        """Initialize the AI Content Polisher."""
        self.config = config
        self.tokenizer = tokenizer
        self.client = None
        
        # Initialize Gemini client for polishing
        if self.config.gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.config.gemini_api_key)
                self.client = genai.GenerativeModel(self.config.ai_polish_model)
                logger.info("AIContentPolisher initialized with Gemini API")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client for polishing: {e}")
                self.client = None
        else:
            logger.warning("No Gemini API key provided for content polishing")
    
    async def polish_content(self, user_query: str, search_results: List['SearchResult'], 
                           crawled_content: List['CrawledContent']) -> Optional[str]:
        """
        Polish extracted content and generate comprehensive report.
        
        Args:
            user_query: Original user research query
            search_results: List of search results
            crawled_content: List of successfully crawled content
            
        Returns:
            Polished markdown report or None if disabled/failed
        """
        if not self.config.ai_polish_content or not self.client:
            return None
        
        # Filter successful content only
        successful_content = [content for content in crawled_content if content.success]
        
        if not successful_content:
            logger.warning("No successful content to polish")
            return None
        
        logger.info(f"Starting AI content polishing with {len(successful_content)} sources")
        
        try:
            # Determine polishing level
            if self.config.ai_polish_content == 2:
                return await self._generate_comprehensive_report(user_query, search_results, successful_content)
            else:
                return await self._basic_content_polish(user_query, successful_content)
        
        except Exception as e:
            logger.error(f"AI content polishing failed: {e}")
            return None
    
    async def _generate_comprehensive_report(self, user_query: str, search_results: List['SearchResult'], 
                                           content: List['CrawledContent']) -> str:
        """Generate comprehensive blog-ready research report."""
        
        # Prepare content summary for the prompt
        content_summary = self._prepare_content_summary(content, max_tokens=4000)
        source_urls = [c.url for c in content]
        
        system_prompt = self._get_comprehensive_report_system_prompt(user_query)
        user_prompt = self._get_comprehensive_report_user_prompt(
            user_query, content_summary, source_urls
        )
        
        try:
            response = await self._call_gemini_api(system_prompt, user_prompt)
            logger.info("Comprehensive report generated successfully")
            return response
        
        except Exception as e:
            logger.error(f"Failed to generate comprehensive report: {e}")
            return None
    
    async def _basic_content_polish(self, user_query: str, content: List['CrawledContent']) -> str:
        """Basic content polishing and cleanup."""
        
        content_text = self._prepare_content_summary(content, max_tokens=3000)
        
        system_prompt = """You are an expert content editor and researcher. Your task is to polish and clean up extracted web content while preserving accuracy and important information."""
        
        user_prompt = f"""Please polish and clean up this research content about: "{user_query}"

EXTRACTED CONTENT:
{content_text}

Please:
1. Remove any remaining formatting issues or noise
2. Organize the information logically
3. Fix grammar and readability
4. Preserve all important facts and data
5. Keep the content factual and well-structured

Return clean, polished markdown content."""
        
        try:
            response = await self._call_gemini_api(system_prompt, user_prompt)
            logger.info("Basic content polishing completed")
            return response
        
        except Exception as e:
            logger.error(f"Failed to polish content: {e}")
            return None
    
    def _get_comprehensive_report_system_prompt(self, user_query: str = "") -> str:
        """Get system prompt for comprehensive report generation with customizable prompts."""
        
        # Try to use custom system prompt from prompts directory
        if DEFAULT_SYSTEM_PROMPT:
            system_prompt = DEFAULT_SYSTEM_PROMPT
            
            # Replace placeholders
            system_prompt = system_prompt.replace("__RESEARCH_QUERY__", user_query)
            system_prompt = system_prompt.replace("__DATE_PLACEHOLDER__", datetime.now().strftime("%Y-%m-%d"))
            
            logger.info("Using custom system prompt from prompts/research_analysis_reporter.py")
            return system_prompt
        else:
            # Fallback to built-in system prompt
            logger.warning("Using fallback system prompt - consider setting up prompts directory")
            return """You are an expert research analyst and technical writer specializing in comprehensive research reports. Your expertise includes:

- Analyzing complex information from multiple sources
- Synthesizing findings into coherent narratives
- Creating engaging, blog-ready content
- Maintaining factual accuracy while improving readability
- Structuring content for maximum impact and understanding

Your task is to transform raw research data into a comprehensive, professional report that would be suitable for publication as a high-quality blog post or research article.

IMPORTANT GUIDELINES:
1. **Accuracy First**: Never fabricate or invent information not present in the sources
2. **Clear Structure**: Use proper markdown headers, lists, and formatting
3. **Engaging Style**: Write in an engaging but professional tone
4. **Comprehensive Coverage**: Address all major aspects of the topic
5. **Source Attribution**: Reference sources appropriately
6. **Blog-Ready**: Format for online publication with good readability
7. **SEO-Friendly**: Use descriptive headers and well-organized content
8. **International Audience**: Make content accessible to global readers"""
    
    def _get_comprehensive_report_user_prompt(self, user_query: str, content_summary: str, source_urls: List[str]) -> str:
        """Get user prompt for comprehensive report generation."""
        
        sources_list = "\n".join([f"- {url}" for url in source_urls[:10]])  # Limit to top 10
        
        return f"""Please create a comprehensive research report about: **"{user_query}"**

## RESEARCH DATA GATHERED:
{content_summary}

## SOURCES ANALYZED:
{sources_list}

## REPORT REQUIREMENTS:

Create a comprehensive, blog-ready markdown report that includes:

### 1. **Executive Summary**
- Brief overview of key findings
- Main insights and takeaways

### 2. **Detailed Analysis**
- Comprehensive coverage of all major aspects
- Data-driven insights where available
- Multiple perspectives on the topic

### 3. **Key Findings & Insights**
- Highlight the most important discoveries
- Identify trends, patterns, or surprising findings
- Present facts and evidence clearly

### 4. **Implications & Impact**
- What do these findings mean?
- How might this affect relevant stakeholders?
- Future considerations or predictions

### 5. **Conclusion**
- Synthesize the main points
- Provide clear concluding thoughts
- Suggest areas for further research if appropriate

## FORMATTING REQUIREMENTS:
- Use proper markdown formatting with headers (##, ###)
- Include bullet points and numbered lists where appropriate
- Make it engaging and readable
- Ensure professional tone suitable for publication
- Target length: 1500-3000 words
- Include relevant data, statistics, or quotes from sources when available

## QUALITY STANDARDS:
- Factual accuracy is paramount
- Clear, engaging writing style
- Logical flow and organization
- Professional presentation
- Ready for blog publication

Transform the raw research data into a polished, comprehensive report that provides genuine value to readers interested in this topic."""
    
    def _prepare_content_summary(self, content: List['CrawledContent'], max_tokens: int = 4000) -> str:
        """Prepare content summary staying within token limits."""
        
        # Prioritize content by length and quality
        sorted_content = sorted(content, key=lambda x: x.content_length, reverse=True)
        
        combined_text = ""
        current_tokens = 0
        
        for item in sorted_content:
            # Estimate tokens for this content piece
            if self.tokenizer:
                item_tokens = self.tokenizer.count_tokens(item.content)
            else:
                item_tokens = len(item.content) // 4  # Rough estimate
            
            # Check if we can add this content
            if current_tokens + item_tokens > max_tokens:
                # Try to add a portion of the content
                remaining_tokens = max_tokens - current_tokens
                remaining_chars = remaining_tokens * 4  # Rough estimate
                
                if remaining_chars > 200:  # Only add if meaningful content
                    partial_content = item.content[:remaining_chars] + "...\n\n"
                    combined_text += f"## Source: {item.url}\n{partial_content}"
                break
            else:
                combined_text += f"## Source: {item.url}\n{item.content}\n\n"
                current_tokens += item_tokens
        
        return combined_text
    
    async def _call_gemini_api(self, system_prompt: str, user_prompt: str) -> str:
        """Make API call to Gemini for content polishing."""
        
        if not self.client:
            raise ValueError("Gemini client not initialized")
        
        try:
            # Prepare the full prompt
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Make the API call
            import google.generativeai as genai
            generation_config = genai.types.GenerationConfig(
                temperature=0.3,  # Lower temperature for more focused output
                top_p=0.8,
                top_k=40,
                max_output_tokens=self.config.max_polish_report_tokens,
            )
            
            response = self.client.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            if response and hasattr(response, 'text'):
                return response.text
            else:
                logger.error("No text response from Gemini API")
                return ""
        
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise


if __name__ == "__main__":
    # Run tests when script is executed directly
    test_tokenizer()