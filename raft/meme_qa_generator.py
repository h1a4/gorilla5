import json
import argparse
import os
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
import math
from collections import defaultdict


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Configuration ----
MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"
MODES = ['base', 'caption', 'both']

# Distractor document ratios (automatically applied to all QA pairs)
DISTRACTOR_RATIOS = [0, 20, 40, 60, 80, 100]
NUM_DISTRACTORS = 3  # Default number of distractor documents

# Category descriptions
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

# 'base' mode prompt template (direct image analysis + document text)
BASE_USER_PROMPT_TEMPLATE = """
Context documents:
{context_docs}

Meme title: {title}
Mode: base (direct image analysis)
Keyword: {keyword}

Question type: {category}
Question type description: {category_description}

Analyze the provided image and document text to generate a CoT-style **multiple-choice question** and **answer**.
- The multiple-choice question must include four options: A, B, C, D.
- Wrap your reasoning process with ##begin_reason## ... ##end_reason##, and in your reasoning clearly explain which documents (oracle or distractor) are relevant to solving the question.
- Present the final answer in the format '<ANSWER>: X' where X is one of A, B, C, D.

Note: Among the provided documents, there may be "oracle documents" that help understand the meme and "distractor documents" that are not relevant.
"""

# 'caption' mode prompt template (caption + document text)
CAPTION_USER_PROMPT_TEMPLATE = """
Context documents:
{context_docs}

Meme title: {title}
Mode: caption (using image caption)
Keyword: {keyword}

Image caption: {caption_text}
Question type: {category}
Question type description: {category_description}

Based on the document text and image caption, generate a CoT-style **multiple-choice question** and **answer**.
- The multiple-choice question must include four options: A, B, C, D.
- Wrap your reasoning process with ##begin_reason## ... ##end_reason##, and in your reasoning clearly explain which documents (oracle or distractor) are relevant to solving the question.
- Present the final answer in the format '<ANSWER>: X' where X is one of A, B, C, D.

Note: Among the provided documents, there may be "oracle documents" that help understand the meme and "distractor documents" that are not relevant.
"""

# 'both' mode prompt template (direct image + caption + document text)
BOTH_USER_PROMPT_TEMPLATE = """
Context documents:
{context_docs}

Meme title: {title}
Mode: both (direct image analysis + caption)
Keyword: {keyword}

Image caption: {caption_text}
Question type: {category}
Question type description: {category_description}

Analyze the provided image, image caption, and document text to generate a CoT-style **multiple-choice question** and **answer**.
- The multiple-choice question must include four options: A, B, C, D.
- Wrap your reasoning process with ##begin_reason## ... ##end_reason##, and in your reasoning clearly explain which documents (oracle or distractor) are relevant to solving the question.
- Present the final answer in the format '<ANSWER>: X' where X is one of A, B, C, D.

Note: Among the provided documents, there may be "oracle documents" that help understand the meme and "distractor documents" that are not relevant.
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
            logger.error(f"Error generating image caption: {e}")
            return "Unable to generate image caption."

# Function to encode image to Base64
def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        return None

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Path to input JSON file")
    parser.add_argument("--documents_root", type=Path, required=False, default=Path("./documents"),
                        help="Root folder path containing category-specific distractor documents")
    parser.add_argument("--output", type=Path, default=Path("./qa_dataset.json"), help="Path to output dataset")
    parser.add_argument("--openai_key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--model", type=str, default="gpt-4-mini", help="Model to use")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    return parser.parse_args()

def load_memes(file_path):
    """Load meme data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_category_distractor_documents(root_folder, category):
    """
    Load distractor documents matching the given category.
    Find documents in category-specific folders.
    """
    category_folder = root_folder / category
    
    # Use root folder if category folder doesn't exist
    if not category_folder.exists():
        logger.warning(f"No folder for category '{category}'. Using root folder.")
        category_folder = root_folder
    
    distractor_docs = []
    
    # Find .txt files
    txt_files = list(category_folder.glob('*.txt'))
    if not txt_files:
        txt_files = list(category_folder.glob('**/*.txt'))
    
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                distractor_docs.append({
                    'path': str(file_path),
                    'content': content,
                    'category': category
                })
        except Exception as e:
            logger.error(f"Error loading text file '{file_path}': {e}")
    
    logger.info(f"Loaded {len(distractor_docs)} distractor documents for category '{category}'")
    return distractor_docs

def ensure_equal_context_length(docs, target_length=None):
    """
    Ensure all document contexts have equal length
    Adjust all document lengths to match the longest document
    """
    if not docs:
        return docs
    
    # Use the longest document if target length not specified
    if target_length is None:
        max_length = max(len(doc['content']) for doc in docs)
        target_length = max_length
    
    adjusted_docs = []
    for doc in docs:
        content = doc['content']
        # Pad with spaces if shorter
        if len(content) < target_length:
            padding = ' ' * (target_length - len(content))
            content = content + padding
        # Truncate if longer
        elif len(content) > target_length:
            content = content[:target_length]
        
        doc_copy = doc.copy()
        doc_copy['content'] = content
        adjusted_docs.append(doc_copy)
    
    return adjusted_docs

def prepare_context_documents(oracle_doc, distractor_docs, has_oracle=True):
    """
    Prepare context documents (based on exact oracle inclusion)
    
    has_oracle: True to include oracle document, False to exclude
    """
    context_docs = []
    total_docs_count = NUM_DISTRACTORS + 1
    
    if len(distractor_docs) < NUM_DISTRACTORS:
        logger.warning(f"Not enough distractor documents. Needed: {NUM_DISTRACTORS}, Available: {len(distractor_docs)}")
        if len(distractor_docs) == 0:
            # Fill all with dummy documents if no distractors available
            context_docs.append({
                'id': 0,
                'is_oracle': has_oracle,
                'content': oracle_doc if has_oracle else "This document is a distractor document not needed for analysis."
            })
            
            for i in range(1, total_docs_count):
                context_docs.append({
                    'id': i,
                    'is_oracle': False,
                    'content': f"This document is a distractor document not needed for analysis. Document ID: {i}"
                })
            
            # Format documents
            formatted_docs = ""
            for i, doc in enumerate(context_docs):
                formatted_docs += f"<DOCUMENT id={i}>\n{doc['content']}\n</DOCUMENT>\n\n"
            
            return {
                'formatted_docs': formatted_docs,
                'docs': context_docs,
                'has_oracle': has_oracle
            }
    
    # Process based on oracle inclusion flag
    if has_oracle:
        # Include oracle document
        context_docs.append({
            'id': 0,
            'is_oracle': True,
            'content': oracle_doc
        })
        
        # Add distractor documents
        distractors_needed = total_docs_count - 1
        selected_distractors = random.sample(distractor_docs, min(distractors_needed, len(distractor_docs)))
        for i, doc in enumerate(selected_distractors):
            context_docs.append({
                'id': i + 1,
                'is_oracle': False,
                'content': doc['content']
            })
    else:
        # Exclude oracle document, use only distractors
        selected_distractors = random.sample(distractor_docs, min(total_docs_count, len(distractor_docs)))
        for i, doc in enumerate(selected_distractors):
            context_docs.append({
                'id': i,
                'is_oracle': False,
                'content': doc['content']
            })
    
    # Fill with dummy documents if not enough distractors
    while len(context_docs) < total_docs_count:
        context_docs.append({
            'id': len(context_docs),
            'is_oracle': False,
            'content': f"This document is a distractor document not needed for analysis. Document ID: {len(context_docs)}"
        })
    
    # Ensure equal document lengths
    context_docs = ensure_equal_context_length(context_docs)
    
    # Shuffle document order
    random.shuffle(context_docs)
    
    # Format documents
    formatted_docs = ""
    for i, doc in enumerate(context_docs):
        formatted_docs += f"<DOCUMENT id={i}>\n{doc['content']}\n</DOCUMENT>\n\n"
    
    return {
        'formatted_docs': formatted_docs,
        'docs': context_docs,
        'has_oracle': has_oracle
    }

def prepare_oracle_distribution(memes, category_docs_mapping):
    """
    Pre-determine oracle document inclusion with exact ratios
    
    returns: {ratio: {category: {mode: {'entries': [entry list], 'has_oracle': [True/False list]}}}}
    """
    # Group memes by category and mode
    category_mode_memes = defaultdict(lambda: defaultdict(list))
    
    for meme in memes:
        for category in CATEGORY_DESCRIPTIONS.keys():
            if category in category_docs_mapping and category_docs_mapping[category]:
                for mode in MODES:
                    category_mode_memes[category][mode].append(meme)
    
    # Determine oracle inclusion for each ratio
    oracle_distribution = {}
    
    for ratio in DISTRACTOR_RATIOS:
        oracle_distribution[ratio] = {}
        
        for category, mode_memes in category_mode_memes.items():
            oracle_distribution[ratio][category] = {}
            
            for mode, entries in mode_memes.items():
                num_entries = len(entries)
                
                if ratio == 0:
                    # 0% ratio: include Oracle in all items
                    has_oracle = [True] * num_entries
                elif ratio == 100:
                    # 100% ratio: exclude Oracle from all items
                    has_oracle = [False] * num_entries
                else:
                    # Middle ratios: exclude Oracle from ratio% of items
                    num_without_oracle = int(num_entries * (ratio / 100))
                    has_oracle = [True] * num_entries
                    
                    # Randomly select items to exclude Oracle
                    indices_without_oracle = random.sample(range(num_entries), num_without_oracle)
                    for idx in indices_without_oracle:
                        has_oracle[idx] = False
                
                oracle_distribution[ratio][category][mode] = {
                    'entries': entries,
                    'has_oracle': has_oracle
                }
    
    return oracle_distribution

def generate_qa_for_entry(entry, captioner, chat_completer, model, category_docs_mapping, oracle_distribution=None):
    """
    Generate QA pairs for each meme entry
    
    oracle_distribution: oracle inclusion information in the format {ratio: {category: {mode: [has_oracle list]}}}
    """
    try:
        # Generate caption
        caption_text = captioner.caption(entry['image_path'])
        oracle_doc = entry['doc_text']
        
        # Encode image (for base and both modes)
        base64_image = encode_image_to_base64(entry['image_path'])
        if not base64_image:
            logger.error(f"Cannot encode image for meme {entry['title']}. Skipping.")
            return []
        
        results = []
        for category, question_pool in CATEGORY_DESCRIPTIONS.items():
            category_description = random.choice(question_pool)
            
            # Get distractor documents for this category
            distractor_docs = category_docs_mapping.get(category, [])
            if not distractor_docs:
                logger.warning(f"No distractor documents for category '{category}'")
                continue
            
            for mode in MODES:
                for ratio in DISTRACTOR_RATIOS:
                    # Check oracle inclusion
                    has_oracle = True  # Default value
                    if oracle_distribution and ratio in oracle_distribution:
                        if category in oracle_distribution[ratio]:
                            if mode in oracle_distribution[ratio][category]:
                                # Find the current entry's index and check oracle inclusion
                                entry_idx = oracle_distribution[ratio][category][mode]['entries'].index(entry)
                                has_oracle = oracle_distribution[ratio][category][mode]['has_oracle'][entry_idx]
                    
                    # Prepare context documents according to ratio
                    context_data = prepare_context_documents(
                        oracle_doc, 
                        distractor_docs,
                        has_oracle=has_oracle
                    )
                    
                    try:
                        # Process by mode
                        if mode == 'base':
                            # 'base' mode: direct image + document text
                            user_prompt = BASE_USER_PROMPT_TEMPLATE.format(
                                context_docs=context_data['formatted_docs'],
                                title=entry['title'],
                                keyword=entry['keyword'],
                                category=category,
                                category_description=category_description
                            )
                            
                            # Image + text
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
                            # 'caption' mode: caption + document text
                            user_prompt = CAPTION_USER_PROMPT_TEMPLATE.format(
                                context_docs=context_data['formatted_docs'],
                                title=entry['title'],
                                keyword=entry['keyword'],
                                caption_text=caption_text,
                                category=category,
                                category_description=category_description
                            )
                            
                            # Text only
                            messages = [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt}
                            ]
                        
                        else:  # 'both' mode
                            # 'both' mode: direct image + caption + document text
                            user_prompt = BOTH_USER_PROMPT_TEMPLATE.format(
                                context_docs=context_data['formatted_docs'],
                                title=entry['title'],
                                keyword=entry['keyword'],
                                caption_text=caption_text,
                                category=category,
                                category_description=category_description
                            )
                            
                            # Image + text
                            messages = [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": [
                                    {"type": "text", "text": user_prompt},
                                    {"type": "image_url", "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }}
                                ]}
                            ]
                        
                        # GPT call - same model for all modes
                        resp = chat_completer(model=model, messages=messages, max_tokens=1024, temperature=0.7)
                        
                        # Process results
                        content = resp.choices[0].message.content.strip()
                        
                        # Extract answer in <ANSWER>: format
                        parts = content.split('<ANSWER>:')
                        question_with_options = parts[0].strip()
                        answer = parts[1].strip() if len(parts) > 1 else ''
                        
                        # Extract reasoning process
                        reason_parts = question_with_options.split('##begin_reason##')
                        question_part = reason_parts[0].strip()
                        reasoning = ""
                        if len(reason_parts) > 1 and '##end_reason##' in reason_parts[1]:
                            reasoning = reason_parts[1].split('##end_reason##')[0].strip()
                        
                        # Save results
                        results.append({
                            "id": str(uuid.uuid4()),
                            "title": entry['title'],
                            "keyword": entry['keyword'],
                            "image_path": entry['image_path'],
                            "mode": mode,
                            "category": category,
                            "distractor_ratio": ratio,
                            "has_oracle": context_data['has_oracle'],
                            "question_with_options": question_part,
                            "reasoning": reasoning,
                            "cot_answer": "<ANSWER>: " + answer,
                            "oracle_doc": oracle_doc,
                            "context_docs": context_data['docs'],
                            "caption": caption_text
                        })
                        
                        logger.info(f"Completed QA generation for meme '{entry['title']}' mode '{mode}' category '{category}' distractor ratio '{ratio}%'")
                        
                    except Exception as e:
                        logger.error(f"Error processing meme {entry['title']} (mode: {mode}, category: {category}, ratio: {ratio}%): {e}")
    
        return results
    except Exception as e:
        logger.error(f"Error processing meme {entry.get('title', 'unknown')}: {e}")
        return []

def main():
    """Main execution function"""
    args = get_args()
    
    # Set up OpenAI client and captioner
    openai_client = build_openai_client(api_key=args.openai_key)
    chat_completer = ChatCompleter(openai_client)
    captioner = Captioner()
    
    # Load meme data
    memes = load_memes(args.input)
    logger.info(f"Loaded {len(memes)} meme entries")
    
    # Check if documents_root exists
    if not args.documents_root.exists():
        logger.warning(f"Documents root folder {args.documents_root} does not exist. Creating dummy documents.")
        os.makedirs(args.documents_root, exist_ok=True)
    
    # Load category-specific distractor documents
    category_docs_mapping = {}
    for category in CATEGORY_DESCRIPTIONS.keys():
        category_docs = load_category_distractor_documents(args.documents_root, category)
        # If no documents found, create dummy documents
        if not category_docs:
            logger.warning(f"No documents found for category {category}. Creating dummy documents.")
            category_docs = [
                {
                    'path': f"dummy_{category}_{i}.txt",
                    'content': f"This is a dummy distractor document for category {category}. Document ID: {i}",
                    'category': category
                } for i in range(NUM_DISTRACTORS * 2)  # Create enough dummy documents
            ]
        category_docs_mapping[category] = category_docs
    
    total_docs = sum(len(docs) for docs in category_docs_mapping.values())
    logger.info(f"Loaded {total_docs} distractor documents across {len(category_docs_mapping)} categories")
    
    # Pre-determine oracle document inclusion with exact ratios
    oracle_distribution = prepare_oracle_distribution(memes, category_docs_mapping)
    logger.info("Determined oracle document inclusion ratios")
    
    # Generate QA pairs for each meme
    dataset = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(generate_qa_for_entry, meme, captioner, chat_completer, 
                           args.model, category_docs_mapping, oracle_distribution)
            for meme in memes
        ]
        
        with tqdm(total=len(futures), desc="Generating meme QA pairs") as pbar:
            for future in as_completed(futures):
                try:
                    results = future.result()
                    dataset.extend(results)
                    pbar.update(1)
                    pbar.set_postfix({'QA pairs': len(dataset)})
                except Exception as e:
                    logger.error(f"Error generating QA: {e}")
    
    logger.info(f"Generated {len(dataset)} QA pairs total")
    
    # Save dataset files for each ratio
    output_dir = args.output.parent
    for ratio in DISTRACTOR_RATIOS:
        ratio_dataset = [item for item in dataset if item['distractor_ratio'] == ratio]
        if ratio_dataset:
            output_path = args.output.with_name(f"{args.output.stem}_ratio_{ratio}{args.output.suffix}")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(ratio_dataset, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(ratio_dataset)} items with distractor ratio {ratio}% to {output_path}")
    
    # Save complete dataset
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved complete dataset to {args.output}")
    
    # Output statistics
    modes_count = {mode: sum(1 for item in dataset if item['mode'] == mode) for mode in MODES}
    categories_count = {cat: sum(1 for item in dataset if item['category'] == cat) for cat in CATEGORY_DESCRIPTIONS.keys()}
    ratios_count = {ratio: sum(1 for item in dataset if item['distractor_ratio'] == ratio) for ratio in DISTRACTOR_RATIOS}
    oracle_count = sum(1 for item in dataset if item['has_oracle'])
    
    logger.info("===== Generated QA Pair Statistics =====")
    logger.info(f"Total memes: {len(memes)}")
    logger.info(f"Total QA pairs: {len(dataset)}")
    logger.info(f"Average QA pairs per meme: {len(dataset)/len(memes):.2f}")
    logger.info(f"Mode distribution: {modes_count}")
    logger.info(f"Category distribution: {categories_count}")
    logger.info(f"Distractor ratio distribution: {ratios_count}")
    logger.info(f"Oracle document inclusion: {oracle_count} ({oracle_count/len(dataset)*100:.2f}%)")

if __name__ == "__main__":
    main()