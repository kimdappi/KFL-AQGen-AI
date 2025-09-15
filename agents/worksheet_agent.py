"""
Worksheet Agent: Generates worksheets following the JSON schema and outputs PDF
"""
import json
import uuid
from typing import Dict, List, Any
from datetime import datetime
from agents.kpop_agent import KpopAgent
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import black, blue, red


class WorksheetAgent:
    def __init__(self):
        self.name = "Worksheet Generator Agent"
        self.kpop_agent = KpopAgent()
        self.schema_path = "data/schemas/worksheet_schema.json"
        self.difficulty_schema_path = "data/schemas/difficulty_levels.json"
    
    def load_schema(self) -> Dict[str, Any]:
        """Load the worksheet schema as SSOT"""
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_difficulty_levels(self) -> Dict[str, Any]:
        """Load difficulty level definitions"""
        with open(self.difficulty_schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def generate_worksheet(self, 
                          topic: str, 
                          difficulty: str, 
                          user_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a complete worksheet following the schema
        """
        # Load schemas
        schema = self.load_schema()
        difficulty_levels = self.load_difficulty_levels()
        
        # Get K-pop information
        kpop_info = self.kpop_agent.search_kpop_info(topic, user_info)
        kpop_sentences = self.kpop_agent.generate_kpop_sentences(kpop_info, difficulty)
        
        # Create worksheet following schema
        worksheet = {
            "id": str(uuid.uuid4()),
            "title": f"K-pop 기반 한국어 학습 문제지 - {topic}",
            "difficulty": difficulty,
            "target_audience": {
                "nationality": user_info.get("nationality", "Unknown"),
                "age_range": difficulty_levels[difficulty]["target_age"],
                "gender": user_info.get("gender", "Any")
            },
            "sections": self._generate_sections(kpop_sentences, difficulty, kpop_info),
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "estimated_time": "30-45 minutes",
                "total_points": 100
            }
        }
        
        return worksheet
    
    def _generate_sections(self, 
                          kpop_sentences: List[str], 
                          difficulty: str, 
                          kpop_info: List[Any]) -> List[Dict[str, Any]]:
        """Generate worksheet sections based on K-pop content"""
        sections = []
        
        # Reading section
        reading_section = {
            "section_id": "reading_1",
            "section_type": "reading",
            "title": "K-pop 관련 읽기",
            "instructions": "다음 K-pop 관련 문장을 읽고 질문에 답하세요.",
            "items": []
        }
        
        for i, sentence in enumerate(kpop_sentences[:2]):
            item = {
                "item_id": f"reading_item_{i+1}",
                "item_type": "short_answer",
                "question": f"다음 문장의 주제는 무엇인가요? '{sentence}'",
                "options": [],
                "correct_answer": "K-pop 관련 내용",
                "explanation": "이 문장은 K-pop 아티스트나 노래에 대한 설명입니다.",
                "difficulty_points": 20,
                "kpop_context": {
                    "artist": kpop_info[i].artist if i < len(kpop_info) else "Unknown",
                    "song": kpop_info[i].song if i < len(kpop_info) else "Unknown",
                    "concept": kpop_info[i].concept if i < len(kpop_info) else "Unknown"
                }
            }
            reading_section["items"].append(item)
        
        sections.append(reading_section)
        
        # Grammar section
        grammar_section = {
            "section_id": "grammar_1",
            "section_type": "grammar",
            "title": "문법 문제",
            "instructions": "다음 문장에서 문법 오류를 찾아 고치세요.",
            "items": []
        }
        
        # Create grammar questions based on difficulty
        if difficulty == "easy":
            grammar_item = {
                "item_id": "grammar_item_1",
                "item_type": "multiple_choice",
                "question": "다음 중 올바른 문장은?",
                "options": [
                    "BTS는 유명한 그룹이다.",
                    "BTS는 유명한 그룹이이다.",
                    "BTS는 유명한 그룹입니다.",
                    "BTS는 유명한 그룹이입니다."
                ],
                "correct_answer": "BTS는 유명한 그룹입니다.",
                "explanation": "존댓말 '-습니다'를 사용한 정확한 문장입니다.",
                "difficulty_points": 15,
                "kpop_context": {
                    "artist": "BTS",
                    "song": "Dynamite",
                    "concept": "basic grammar"
                }
            }
        else:
            grammar_item = {
                "item_id": "grammar_item_1",
                "item_type": "fill_blank",
                "question": "다음 문장을 완성하세요: 'BLACKPINK의 새 앨범은 많은 팬들____ 기대를 모으고 있다.'",
                "options": ["에게", "에게서", "로부터", "에서"],
                "correct_answer": "에게",
                "explanation": "'에게'는 동작의 대상이나 방향을 나타내는 조사입니다.",
                "difficulty_points": 25,
                "kpop_context": {
                    "artist": "BLACKPINK",
                    "song": "How You Like That",
                    "concept": "advanced grammar"
                }
            }
        
        grammar_section["items"].append(grammar_item)
        sections.append(grammar_section)
        
        return sections
    
    def generate_pdf(self, worksheet: Dict[str, Any], output_path: str) -> str:
        """
        Generate PDF from worksheet JSON data
        """
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title style
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1,  # Center alignment
            textColor=blue
        )
        
        # Section title style
        section_style = ParagraphStyle(
            'SectionTitle',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=black
        )
        
        # Question style
        question_style = ParagraphStyle(
            'Question',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            leftIndent=20
        )
        
        # Answer style
        answer_style = ParagraphStyle(
            'Answer',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=12,
            leftIndent=40,
            textColor=red
        )
        
        # Add title
        story.append(Paragraph(worksheet['title'], title_style))
        story.append(Spacer(1, 12))
        
        # Add metadata
        metadata = worksheet.get('metadata', {})
        story.append(Paragraph(f"난이도: {worksheet['difficulty']}", styles['Normal']))
        story.append(Paragraph(f"예상 소요 시간: {metadata.get('estimated_time', '30-45분')}", styles['Normal']))
        story.append(Paragraph(f"총 점수: {metadata.get('total_points', 100)}점", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Add sections
        for section in worksheet.get('sections', []):
            story.append(Paragraph(section['title'], section_style))
            story.append(Paragraph(section['instructions'], styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Add items
            for i, item in enumerate(section.get('items', []), 1):
                story.append(Paragraph(f"{i}. {item['question']}", question_style))
                
                # Add options for multiple choice
                if item.get('options') and len(item['options']) > 0:
                    for j, option in enumerate(item['options']):
                        story.append(Paragraph(f"   {chr(65+j)}. {option}", styles['Normal']))
                
                # Add K-pop context if available
                kpop_context = item.get('kpop_context', {})
                if kpop_context.get('artist'):
                    story.append(Paragraph(
                        f"<i>K-pop 컨텍스트: {kpop_context['artist']} - {kpop_context.get('song', 'N/A')}</i>", 
                        styles['Italic']
                    ))
                
                story.append(Spacer(1, 8))
            
            story.append(Spacer(1, 20))
        
        # Add answer key section
        story.append(PageBreak())
        story.append(Paragraph("정답 및 해설", title_style))
        story.append(Spacer(1, 20))
        
        for section in worksheet.get('sections', []):
            story.append(Paragraph(section['title'] + " - 정답", section_style))
            
            for i, item in enumerate(section.get('items', []), 1):
                story.append(Paragraph(f"{i}. 정답: {item['correct_answer']}", answer_style))
                if item.get('explanation'):
                    story.append(Paragraph(f"해설: {item['explanation']}", styles['Normal']))
                story.append(Spacer(1, 8))
            
            story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
        return output_path
