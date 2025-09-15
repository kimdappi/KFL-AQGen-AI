# Korean Worksheet Generator

A simplified Korean language worksheet generator that uses K-pop content to create educational materials. The system consists of three specialized agents working together to generate, validate, and ensure quality worksheets.

## Structure

```
.
├── main.py                    # Main orchestration script
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── data/
│   ├── pdfs/                 # PDF worksheets by difficulty
│   │   ├── easy/            # Beginner level PDFs 
│   │   ├── medium/          # Intermediate level PDFs
│   │   └── hard/            # Advanced level PDFs
│   └── schemas/             # JSON schemas (SSOT)
│       ├── worksheet_schema.json      # Worksheet structure definition
│       └── difficulty_levels.json     # Difficulty level specifications
└── agents/
    ├── kpop_agent.py         # K-pop information retrieval & sentence generation
    ├── worksheet_agent.py    # Worksheet generation following schema
    └── critic_agent.py       # Quality validation and feedback
```

## Features

- **K-pop Agent**: Uses MCP (Model Context Protocol) to search and retrieve K-pop information, create sentence
- **Worksheet Agent**: Generates worksheets following a strict JSON schema (SSOT)
- **Critic Agent**: Validates generated worksheets for quality, schema compliance, and K-pop relevance
- **Difficulty-based Generation**: Supports difficulty levels
- **Schema-driven**: All worksheets follow a predefined JSON schema for consistency

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Place PDF worksheets** in the appropriate difficulty folders:
   - `data/pdfs/easy/` - Beginner level worksheets
   - `data/pdfs/medium/` - Intermediate level worksheets  
   - `data/pdfs/hard/` - Advanced level worksheets

3. **Run the generator**:
   ```bash
   python main.py
   ```

## How It Works

1. **K-pop Agent** searches for relevant K-pop information using MCP
2. **Worksheet Agent** generates a worksheet following the JSON schema, incorporating K-pop content
3. **Critic Agent** validates the worksheet for:
   - Schema compliance
   - Difficulty appropriateness
   - Content quality
   - K-pop relevance

## Configuration

Edit `config.py` to customize:
- Data paths
- MCP settings
- Validation thresholds
- Generation parameters

## Output

The system generates:
- A complete worksheet in **PDF format** ready for printing/use
- Validation results in JSON format (for debugging/analysis)
- Suggestions for improvement

## Schema (SSOT)

The `data/schemas/worksheet_schema.json` defines the exact structure that all worksheets must follow, ensuring consistency and quality across all generated content.

## Future Enhancements

- Real MCP integration for K-pop data
- PDF parsing and template extraction
- Web-based K-pop information retrieval
- Advanced validation rules
- Multi-language support