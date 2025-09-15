"""
base_agent.py - 모든 에이전트의 기본 클래스
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """모든 에이전트의 기본 클래스"""
    
    def __init__(self, model_name: str = None, agent_name: str = "BaseAgent"):
        """
        BaseAgent 초기화
        
        Args:
            model_name: Hugging Face 모델 이름
            agent_name: 에이전트 식별 이름
        """
        self.agent_name = agent_name
        
        # 기본 모델 설정
        if model_name is None:
            model_name = "skt/kogpt2-base-v2"  # 가벼운 한국어 모델
        
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"🚀 {self.agent_name} 초기화: {model_name}")
        logger.info(f"💻 디바이스: {self.device}")
        
        self._load_model()
    
    def _load_model(self):
        """Hugging Face에서 모델과 토크나이저 로드"""
        try:
            logger.info(f"📥 {self.agent_name} 모델 로드 중...")
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # 파이프라인 생성
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info(f"✅ {self.agent_name} 모델 로드 완료!")
            
        except Exception as e:
            logger.error(f"❌ {self.agent_name} 모델 로드 실패: {e}")
            raise
    
    def generate_response(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
        """
        프롬프트에 대한 응답 생성
        
        Args:
            prompt: 입력 프롬프트
            max_new_tokens: 최대 생성 토큰 수
            temperature: 생성 온도 (창의성 조절)
        
        Returns:
            생성된 응답 텍스트
        """
        try:
            formatted_prompt = self._format_prompt(prompt)
            
            if self.pipeline:
                outputs = self.pipeline(
                    formatted_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_full_text=False
                )
                response = outputs[0]['generated_text']
            else:
                inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=0.9,
                        do_sample=True,
                        repetition_penalty=1.1
                    )
                
                generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            return self._clean_output(response)
            
        except Exception as e:
            logger.error(f"❌ {self.agent_name} 응답 생성 오류: {e}")
            return ""
    
    def _format_prompt(self, prompt: str) -> str:
        """모델별 프롬프트 포맷팅"""
        model_lower = self.model_name.lower()
        
        if 'kogpt' in model_lower:
            return f"질문: {prompt}\n답변:"
        elif 'polyglot' in model_lower:
            return f"### 질문:\n{prompt}\n\n### 응답:\n"
        else:
            return prompt
    
    def _clean_output(self, output: str) -> str:
        """출력 정리"""
        if not output:
            return ""
        
        # 특수 토큰 제거
        special_tokens = ["<|endoftext|>", "</s>", "###", "\n\n\n"]
        for token in special_tokens:
            output = output.replace(token, "")
        
        return output.strip()
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        각 에이전트별 처리 로직 (상속받아 구현)
        
        Args:
            input_data: 입력 데이터 딕셔너리
        
        Returns:
            처리 결과 딕셔너리
        """
        pass
    
    def validate_input(self, input_data: Dict[str, Any], required_fields: list) -> bool:
        """
        입력 데이터 검증
        
        Args:
            input_data: 입력 데이터
            required_fields: 필수 필드 리스트
        
        Returns:
            검증 성공 여부
        """
        for field in required_fields:
            if field not in input_data:
                logger.error(f"❌ {self.agent_name}: 필수 필드 '{field}' 누락")
                return False
        return True