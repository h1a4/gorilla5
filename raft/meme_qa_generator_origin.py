import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import base64
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from client_utils import build_openai_client, ChatCompleter
from tqdm import tqdm
import logging
import uuid
import random


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- 설정 ----
MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"
MODES = ['base', 'caption', 'both']

# 카테고리 및 상세 설명
CATEGORY_DESCRIPTIONS = {
    "Understanding Cultural Context": [
        "What cultural phenomenon does this meme reference?",
        "Which era or generation's culture is reflected in this meme?",
        "What is the social background that led to the popularity of this meme?",
        "What shared experience or perception of a social group does this meme portray?",
        "What cultural message does this meme convey?"
    ],
    "Metaphor and Symbol Interpretation": [
        "What does the specific image or text in this meme symbolize?",
        "What metaphorical meaning is conveyed through the expression in the meme?",
        "What social message is delivered by the symbolic elements of this meme?",
        "What does the visual composition of the meme symbolize?",
        "What concept is expressed metaphorically in the core phrase of this meme?"
    ],
    "Detecting Satire and Irony": [
        "In what way does this meme incorporate satirical elements?",
        "What social phenomenon is being mocked through the expression in this meme?",
        "What kind of irony is revealed in this meme?",
        "What contradictory situation is being highlighted in this meme?",
        "What is the social meaning behind the humor used in this meme?"
    ],
    "Analyzing Social Conflict": [
        "What social conflict or tension is reflected in this meme?",
        "Which social groups are in opposition in this meme?",
        "What structural inequality is exposed by this meme?",
        "How is this meme related to power dynamics or social status?",
        "What ideological clash within society is expressed through this meme?"

    ],
    "Image-Text Integration": [
        "How do the image and text in this meme complement each other's meaning?",
        "What message is conveyed through the combination of image and text in this meme?",
        "What is the relationship between the text and the image in this meme?",
        "How does the meaning differ when the text is seen without the image?",
        "How do the visual and linguistic elements interact in this meme?"

    ],
    "Context-Based Critical Thinking": [
        "What social issue is this meme criticizing?",
        "What potential counterarguments exist to the perspective shown in this meme?",
        "What viewpoint does this meme present on a specific issue?",
        "How does this meme encourage critical thinking in its audience?",
        "What assumptions or biases are hidden within this meme?"

    ]
}

# ---- PROMPT TEMPLATES ----
SYSTEM_PROMPT = (
    "You are a Chain-of-Thought multiple-choice question-answer generator "
    "specialized in analyzing social-conflict memes. "
    "For a given meme context with documents (oracle and distractors), you must produce one multiple-choice question with 4 options (A, B, C, D) and its answer "
    "including explicit reasoning. Surround reasoning with '##begin_reason##' and '##end_reason##', "
    "and mark the final answer as '<ANSWER>: X' where X is the correct option letter."
)

# 'base' 모드용 프롬프트 템플릿 (이미지 직접 전달 + 문서 텍스트)
BASE_USER_PROMPT_TEMPLATE = """
Context document:
<DOCUMENT>{doc_text}</DOCUMENT>

밈 제목: {title}
모드: base (이미지 직접 분석)
키워드: {keyword}

질문 유형: {category}
질문 유형 설명: {category_description}

제공된 이미지와 문서 텍스트를 분석하여, CoT 스타일로 **객관식 질문 하나**와 **답변**을 생성해 주세요.
- 객관식 질문은 A, B, C, D 네 가지 선택지를 포함해야 합니다.
- 추론 과정은 ##begin_reason## ... ##end_reason## 으로 감싸고,
- 최종 답은 '<ANSWER>: X' 형식으로 제시하세요 (X는 A, B, C, D 중 하나).
"""

# 'caption' 모드용 프롬프트 템플릿 (캡션 + 문서 텍스트)
CAPTION_USER_PROMPT_TEMPLATE = """
Context document:
<DOCUMENT>{doc_text}</DOCUMENT>

밈 제목: {title}
모드: caption (이미지 캡션 활용)
키워드: {keyword}

이미지 캡션: {caption_text}
질문 유형: {category}
질문 유형 설명: {category_description}

문서 텍스트와 이미지 캡션을 바탕으로, CoT 스타일로 **객관식 질문 하나**와 **답변**을 생성해 주세요.
- 객관식 질문은 A, B, C, D 네 가지 선택지를 포함해야 합니다.
- 추론 과정은 ##begin_reason## ... ##end_reason## 으로 감싸고,
- 최종 답은 '<ANSWER>: X' 형식으로 제시하세요 (X는 A, B, C, D 중 하나).
"""

# 'both' 모드용 프롬프트 템플릿 (이미지 직접 전달 + 캡션 + 문서 텍스트)
BOTH_USER_PROMPT_TEMPLATE = """
Context document:
<DOCUMENT>{doc_text}</DOCUMENT>

밈 제목: {title}
모드: both (이미지 직접 분석 + 캡션 활용)
키워드: {keyword}

이미지 캡션: {caption_text}
질문 유형: {category}
질문 유형 설명: {category_description}

제공된 이미지, 이미지 캡션, 그리고 문서 텍스트를 모두 분석하여, CoT 스타일로 **객관식 질문 하나**와 **답변**을 생성해 주세요.
- 객관식 질문은 A, B, C, D 네 가지 선택지를 포함해야 합니다.
- 추론 과정은 ##begin_reason## ... ##end_reason## 으로 감싸고,
- 최종 답은 '<ANSWER>: X' 형식으로 제시하세요 (X는 A, B, C, D 중 하나).
"""

# ---- IMAGE CAPTIONING ----
class Captioner:
    def __init__(self):
        self.model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def caption(self, image_path: str) -> str:
        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.feature_extractor(images=[image], return_tensors="pt").pixel_values
            output_ids = self.model.generate(pixel_values)
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"이미지 캡션 생성 중 오류 발생: {e}")
            return "이미지 캡션을 생성할 수 없습니다."

# 이미지를 Base64로 인코딩하는 함수
def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"이미지 인코딩 중 오류 발생: {e}")
        return None

def get_args():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="입력 JSON 파일 경로")
    parser.add_argument("--output", type=Path, default=Path("./qa_dataset.json"), help="출력 데이터셋 경로")
    parser.add_argument("--openai_key", type=str, default=None, help="OpenAI API 키")
    parser.add_argument("--model", type=str, default="gpt-4-mini", help="사용할 모델")
    parser.add_argument("--workers", type=int, default=4, help="병렬 처리 워커 수")
    return parser.parse_args()

def load_memes(file_path):
    """JSON 파일에서 밈 데이터 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_qa_for_entry(entry, captioner, chat_completer, model):
    """각 밈 항목에 대한 QA 쌍 생성"""
    # 캡션 생성
    caption_text = captioner.caption(entry['image_path'])
    doc_text = entry['doc_text']
    
    # 이미지 인코딩 (base와 both 모드에서 사용)
    base64_image = encode_image_to_base64(entry['image_path'])
    if not base64_image:
        logger.error(f"밈 {entry['title']}의 이미지를 인코딩할 수 없습니다. 건너뜁니다.")
        return []
    
    results = []
    for category, question_pool in CATEGORY_DESCRIPTIONS.items():
        category_description = random.choice(question_pool)
        for mode in MODES:
            try:
                # 모드별 처리
                if mode == 'base':
                    # 'base' 모드: 이미지 직접 전달 + 문서 텍스트
                    user_prompt = BASE_USER_PROMPT_TEMPLATE.format(
                        doc_text=doc_text,
                        title=entry['title'],
                        keyword=entry['keyword'],
                        category=category,
                        category_description=category_description
                    )
                    
                    # 이미지 + 텍스트
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }}
                        ]}
                    ]
                
                elif mode == 'caption':
                    # 'caption' 모드: 캡션 + 문서 텍스트
                    user_prompt = CAPTION_USER_PROMPT_TEMPLATE.format(
                        doc_text=doc_text,
                        title=entry['title'],
                        keyword=entry['keyword'],
                        caption_text=caption_text,
                        category=category,
                        category_description=category_description
                    )
                    
                    # 텍스트만
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ]
                
                else:  # 'both' 모드
                    # 'both' 모드: 이미지 직접 전달 + 캡션 + 문서 텍스트
                    user_prompt = BOTH_USER_PROMPT_TEMPLATE.format(
                        doc_text=doc_text,
                        title=entry['title'],
                        keyword=entry['keyword'],
                        caption_text=caption_text,
                        category=category,
                        category_description=category_description
                    )
                    
                    # 이미지 + 텍스트
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }}
                        ]}
                    ]
                
                # GPT 호출 - 모든 모드에서 같은 모델 사용
                resp = chat_completer(model=model, messages=messages, max_tokens=1024, temperature=0.7)
                
                # 결과 처리
                content = resp.choices[0].message.content.strip()
                
                # <ANSWER>: 형식으로 답변 추출
                parts = content.split('<ANSWER>:')
                question_with_options = parts[0].strip()
                answer = parts[1].strip() if len(parts) > 1 else ''
                
                # 추론 과정 추출
                reason_parts = question_with_options.split('##begin_reason##')
                question_part = reason_parts[0].strip()
                reasoning = ""
                if len(reason_parts) > 1 and '##end_reason##' in reason_parts[1]:
                    reasoning = reason_parts[1].split('##end_reason##')[0].strip()
                
                # 결과 저장
                results.append({
                    "id": str(uuid.uuid4()),
                    "title": entry['title'],
                    "keyword": entry['keyword'],
                    "image_path": entry['image_path'],
                    "mode": mode,
                    "category": category,
                    "question_with_options": question_part,
                    "reasoning": reasoning,
                    "cot_answer": "<ANSWER>: " + answer,
                    "doc_text": doc_text,
                    "caption": caption_text
                })
                
                logger.info(f"밈 '{entry['title']}' 모드 '{mode}' 카테고리 '{category}'에 대한 QA 생성 완료")
                
            except Exception as e:
                logger.error(f"밈 {entry['title']} 처리 중 오류 발생 (모드: {mode}, 카테고리: {category}): {e}")
    
    return results

def main():
    """메인 실행 함수"""
    args = get_args()
    
    # OpenAI 클라이언트 및 캡셔너 설정
    openai_client = build_openai_client(api_key=args.openai_key)
    chat_completer = ChatCompleter(openai_client)
    captioner = Captioner()
    
    # 밈 데이터 로드
    memes = load_memes(args.input)
    logger.info(f"{len(memes)}개의 밈 데이터를 로드했습니다.")
    
    # 밈별로 QA 쌍 생성
    dataset = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(generate_qa_for_entry, meme, captioner, chat_completer, args.model)
            for meme in memes
        ]
        
        with tqdm(total=len(futures), desc="밈 QA 생성") as pbar:
            for future in as_completed(futures):
                try:
                    results = future.result()
                    dataset.extend(results)
                    pbar.update(1)
                    pbar.set_postfix({'QA 쌍': len(dataset)})
                except Exception as e:
                    logger.error(f"QA 생성 중 오류 발생: {e}")
    
    logger.info(f"총 {len(dataset)}개의 QA 쌍을 생성했습니다.")
    
    # 데이터셋 저장
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    logger.info(f"데이터셋이 {args.output}에 저장되었습니다.")
    
    # 통계 정보 출력
    modes_count = {mode: sum(1 for item in dataset if item['mode'] == mode) for mode in MODES}
    categories_count = {cat: sum(1 for item in dataset if item['category'] == cat) for cat in CATEGORY_DESCRIPTIONS.keys()}
    
    logger.info("===== 생성된 QA 쌍 통계 =====")
    logger.info(f"총 밈 수: {len(memes)}")
    logger.info(f"총 QA 쌍 수: {len(dataset)}")
    logger.info(f"밈당 평균 QA 쌍 수: {len(dataset)/len(memes):.2f}")
    logger.info(f"모드별 분포: {modes_count}")
    logger.info(f"카테고리별 분포: {categories_count}")

if __name__ == "__main__":
    main()