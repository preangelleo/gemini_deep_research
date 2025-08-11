# System Prompts for AI Content Polishing

This directory contains customizable system prompts used by the Gemini Deep Research Agent's AI Content Polisher. These prompts control the structure, style, and focus of your AI-generated research reports.

## 📋 Quick Customization

### To Change Report Style:
1. **Open**: `research_analysis_reporter.py`
2. **Find**: `DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPT_RESEARCH_ANALYSIS`
3. **Change to**: Any of the available prompt styles
4. **Save** and run your research

### Example:
```python
# For executive summaries instead of comprehensive reports:
DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPT_EXECUTIVE_SUMMARY
```

## 🎨 Available Report Styles

### 1. Comprehensive Research (Default)
- **Best for**: Detailed analysis and blog-ready content
- **Style**: Professional, thorough, well-structured
- **Length**: 2000-8000 words typically
- **Use case**: Research articles, comprehensive analysis

### 2. Executive Summary
- **Best for**: Business decision-makers
- **Style**: Concise, action-oriented, strategic
- **Length**: 500-1000 words
- **Use case**: Board presentations, strategic planning

### 3. Academic Research
- **Best for**: Scholarly work and literature reviews
- **Style**: Formal, methodical, citation-heavy
- **Length**: 1000-5000 words
- **Use case**: Research papers, academic analysis

### 4. Technical Analysis
- **Best for**: Implementation and technical teams
- **Style**: Specification-focused, practical
- **Length**: 1000-3000 words
- **Use case**: Technical documentation, system design

### 5. Market Intelligence
- **Best for**: Business and competitive analysis
- **Style**: Data-driven, strategic, commercial
- **Length**: 1000-2000 words
- **Use case**: Market research, competitive analysis

## ✏️ Creating Custom Prompts

### Prompt Structure:
```python
SYSTEM_PROMPT_YOUR_STYLE = """You are an expert [role]. Your task is to [objective].

[Context and requirements]

## REPORT STRUCTURE AND REQUIREMENTS

### 1. [Section Name]
[Instructions for this section]

### 2. [Another Section]
[More instructions]

## OUTPUT FORMATTING REQUIREMENTS
- **Use valid Markdown formatting throughout**
- **Structure with clear headings**
- [More formatting rules]

## WRITING STYLE GUIDELINES
- **[Style characteristic 1]**
- **[Style characteristic 2]**

## CRITICAL REQUIREMENTS
1. **[Important requirement]**
2. **[Another requirement]**
"""
```

### Key Elements to Customize:

#### 1. **Role Definition**
```python
"You are an expert research analyst..."  # Change this
```

#### 2. **Report Structure**
- Define sections (Introduction, Analysis, Conclusion, etc.)
- Set the order and emphasis
- Specify required elements

#### 3. **Writing Style**
- Tone (formal, casual, technical)
- Audience (executives, researchers, general public)
- Complexity level

#### 4. **Output Format**
- Markdown structure
- Citation style
- Length requirements

### Example Custom Prompt:
```python
SYSTEM_PROMPT_NEWS_SUMMARY = """You are a news editor creating daily briefings. Your task is to create concise news summaries.

## STRUCTURE:
### Headlines (3-5 key stories)
### Analysis (brief context for each)  
### Impact Assessment (who's affected)
### What to Watch (upcoming developments)

## STYLE:
- Clear, journalistic tone
- Maximum 1000 words
- Bullet points preferred
- No speculation beyond facts
"""
```

## 🔧 Advanced Customization

### Placeholder Variables:
The system automatically replaces these in your prompts:
- `__RESEARCH_QUERY__` → User's research question
- `__DATE_PLACEHOLDER__` → Current date (YYYY-MM-DD)

### Example Usage:
```python
SYSTEM_PROMPT_CUSTOM = """Research Focus: __RESEARCH_QUERY__
Report Date: __DATE_PLACEHOLDER__

Your analysis should specifically address the research question above..."""
```

### Domain-Specific Prompts:
Create specialized prompts for different domains:

```python
SYSTEM_PROMPT_MEDICAL = """You are a medical research analyst..."""
SYSTEM_PROMPT_FINANCE = """You are a financial analyst..."""
SYSTEM_PROMPT_TECHNOLOGY = """You are a technology researcher..."""
```

## 📝 Best Practices

### ✅ DO:
- Be specific about structure requirements
- Define the target audience clearly
- Include formatting instructions
- Set clear length guidelines
- Specify citation requirements

### ❌ DON'T:
- Make prompts too long (>2000 words)
- Include contradictory instructions
- Forget markdown formatting requirements
- Leave output structure ambiguous

## 🧪 Testing Your Prompts

### Test Your Custom Prompt:
1. **Save** your changes to `research_analysis_reporter.py`
2. **Run**: `python examples/ai_polishing_demo.py`
3. **Review** the generated report style
4. **Iterate** on the prompt based on results

### Quick Test Topics:
- "Artificial intelligence safety regulations 2024"
- "Renewable energy storage breakthrough" 
- "Remote work productivity trends"

## 📚 Prompt Engineering Tips

### For Better Results:
1. **Be Specific**: "Create 5 sections" vs "Create several sections"
2. **Set Constraints**: Word limits, format requirements
3. **Define Quality**: What makes a "good" report in your domain?
4. **Consider Context**: Who will read this report?
5. **Iterate**: Test and refine based on actual outputs

### Common Issues:
- **Too verbose**: Add length constraints
- **Poor structure**: Define sections explicitly  
- **Wrong tone**: Adjust role and audience description
- **Missing key info**: Add required elements list

## 🚀 Advanced Features

### Conditional Prompts:
```python
def get_custom_prompt(research_domain: str) -> str:
    if "medical" in research_domain.lower():
        return SYSTEM_PROMPT_MEDICAL
    elif "finance" in research_domain.lower():
        return SYSTEM_PROMPT_FINANCE
    else:
        return DEFAULT_SYSTEM_PROMPT
```

### Multi-Language Support:
```python
SYSTEM_PROMPT_SPANISH = """Eres un analista de investigación experto..."""
SYSTEM_PROMPT_CHINESE = """你是一名专业的研究分析师..."""
```

---

## 💡 Need Help?

**Creating a new prompt style?** Start with one of the existing prompts and modify it step by step.

**Not getting the results you want?** Try the interactive demo to test different prompts quickly:
```bash
python examples/ai_polishing_demo.py
```

**Questions or ideas?** The prompt system is designed to be flexible - experiment and customize to match your specific needs!