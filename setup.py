import json
import os
import ollama
import chromadb
import  re
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch




# 모델과 토크나이저 로드
model_name = "monologg/koelectra-base-v3-discriminator"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# DB 및 데이터 파일 경로 설정
DB_PATH = "chroma_db"
JSON_FILE_PATH = os.path.join("app", "data", "tech_regulations.json")

# Persistent ChromaDB 클라이언트 연결 및 "docs" 컬렉션 로드 (없으면 생성)
client = chromadb.PersistentClient(
    path=DB_PATH,
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)
try:
    collection = client.get_collection(name="docs")
except Exception:
    collection = client.create_collection(name="docs")
def get_skeleton_text(record_id: str, json_file_path: str = JSON_FILE_PATH) -> str:
    """
    주어진 record_id (예: "doc_6_chap_2장_sec_8절_art_80조" 또는 "doc_6_art_80조")에 해당하는
    항목의 원본 JSON 콘텐츠를 스켈레톤 형식으로 재구성하여 반환합니다.
    
    스켈레톤 형식 예시:
      문서제목
      발행번호
      (있다면) 제{chapter_number}장 {chapter_title}
        (있다면)   ├─ 제{section_number}절 {section_title}
            ├─ {article_number}조({article_title}): {article_text}
                ├─ {paragraph_symbol} {paragraph_text} (중간은 ├─, 마지막은 └─)
                └─ ... (항목들도 동일하게 들여쓰기)
                
    장과 절 정보가 생략된 경우, 해당 문서 내에서 유일한 해당 조(article)를 검색하여
    소속 장/절 정보를 함께 출력합니다.
    """
    # JSON 파일 로드
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return f"JSON 파일을 로드하는 데 실패했습니다: {e}"
        
    documents = data.get("documents", [])
    
    # ID 파싱
    pattern = r"doc_(?P<doc_id>[^_]+)(?:_chap_(?P<chap>[^_]+))?(?:_sec_(?P<sec>[^_]+))?(?:_art_(?P<art>.+))?"
    m = re.match(pattern, record_id)
    if not m:
        return f"올바르지 않은 ID 형식: {record_id}"
    
    groups = m.groupdict()
    doc_id = groups.get("doc_id")
    chap = groups.get("chap")
    sec = groups.get("sec")
    art = groups.get("art")
    
    # 문서(document) 찾기
    target_doc = None
    for doc in documents:
        if str(doc.get("document_id", "")) == doc_id:
            target_doc = doc
            break
    if target_doc is None:
        return f"문서(document) ID '{doc_id}'를 찾지 못했습니다."
    
    lines = []
    # 문서 레벨 출력
    doc_title = target_doc.get("document_title", "")
    promulgation = target_doc.get("promulgation_number", "")
    lines.append(doc_title)
    lines.append(promulgation)
    
    # Article level 검색 로직
    article_found = None
    found_chapter = None
    found_section = None

    if art is not None:
        # case 1: 장과 절 정보가 모두 제공된 경우 (일반 케이스)
        if chap is not None and sec is not None:
            for chapter in target_doc.get("chapters", []):
                if chapter.get("chapter_number", "") == chap:
                    found_chapter = chapter
                    for section in chapter.get("sections", []):
                        if section.get("section_number", "default") == sec:
                            found_section = section
                            for article in section.get("articles", []):
                                if article.get("article_number", "") == art:
                                    article_found = article
                                    break
                            break
                    break
        # case 2: 장은 제공되었으나 절 정보가 생략된 경우
        elif chap is not None and sec is None:
            for chapter in target_doc.get("chapters", []):
                if chapter.get("chapter_number", "") == chap:
                    found_chapter = chapter
                    for section in chapter.get("sections", []):
                        for article in section.get("articles", []):
                            if article.get("article_number", "") == art:
                                found_section = section
                                article_found = article
                                break
                        if article_found:
                            break
                    break
        # case 3: 장과 절 정보 모두 생략된 경우 -> 전체 문서에서 해당 조(article) 검색 (유일하다고 가정)
        elif chap is None and sec is None:
            for chapter in target_doc.get("chapters", []):
                for section in chapter.get("sections", []):
                    for article in section.get("articles", []):
                        if article.get("article_number", "") == art:
                            found_chapter = chapter
                            found_section = section
                            article_found = article
                            break
                    if article_found:
                        break
                if article_found:
                    break

    # 만약 art가 제공되지 않았다면 (문서, 장, 절 레벨만) 기존 로직 사용
    if art is None:
        # 장 레벨만 제공된 경우
        if chap is not None:
            chapter_found = None
            for chapter in target_doc.get("chapters", []):
                if chapter.get("chapter_number", "") == chap:
                    chapter_found = chapter
                    break
            if chapter_found is None:
                return "\n".join(lines) + f"\n장(chapter) 번호 '{chap}'을(를) 찾지 못했습니다."
            chap_num = chapter_found.get("chapter_number", "")
            if not chap_num.startswith("제"):
                chap_num = "제" + chap_num
            chap_title = chapter_found.get("chapter_title", "")
            lines.append(f"{chap_num} {chap_title}")
            return "\n".join(lines)
        else:
            return "\n".join(lines)
    
    # 만약 조(article)를 찾지 못했다면
    if article_found is None:
        return "\n".join(lines) + f"\n조(article) 번호 '{art}'을(를) 찾지 못했습니다."
    
    # 출력할 때, 만약 found_chapter 및 found_section가 있다면 출력
    if found_chapter is not None:
        chap_num = found_chapter.get("chapter_number", "")
        if not chap_num.startswith("제"):
            chap_num = "제" + chap_num
        chap_title = found_chapter.get("chapter_title", "")
        lines.append(f"{chap_num} {chap_title}")
    if found_section is not None:
        sec_num = found_section.get("section_number", "")
        sec_title = found_section.get("section_title", "")
        if not sec_num.startswith("제"):
            sec_num = "제" + sec_num
        lines.append(f"  ├─ {sec_num} {sec_title}".rstrip())
    
    # 조(article) 레벨 출력
    art_num = article_found.get("article_number", "")
    art_title = article_found.get("article_title", "")
    art_text = article_found.get("article_text", "").strip()
    if not art_num.startswith("제"):
        art_num = "제" + art_num
    article_line = f"      ├─ {art_num}({art_title}): {art_text}".rstrip()
    lines.append(article_line)
    
    # 문단 및 항목
    paragraphs = article_found.get("paragraphs", [])
    for i, para in enumerate(paragraphs):
        p_symbol = para.get("paragraph_symbol", "")
        p_text = para.get("paragraph_text", "").strip()
        branch = "├─" if i < len(paragraphs) - 1 else "└─"
        lines.append(f"          {branch} {p_symbol} {p_text}".rstrip())
        items = para.get("items", [])
        for item in items:
            item_text = item.get("item_text", "").strip()
            lines.append(f"              ├─ {item_text}".rstrip())
    
    return "\n".join(lines)


def generate_embedding(text: str, model: str = "mxbai-embed-large") -> list:
    """
    올라마 임베딩 API를 사용하여 주어진 텍스트의 임베딩을 생성합니다.
    """
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

def ingest_documents(model: str = "mxbai-embed-large"):
    """
    JSON 파일(app/data/tech_regulations.json) 내의 각 문서에 대해,
    - 문서(document) 단위,
    - 장(chapter) 단위,
    - 조(article) 단위
    임베딩을 각각 생성하여 Chroma DB에 저장합니다.
    """
    # 이미 DB에 레코드가 있으면 ingestion을 건너뜁니다.
    if collection.count() > 0:
        print("이미 임베딩이 존재합니다. Ingestion을 건너뜁니다.")
        return

    if not os.path.exists(JSON_FILE_PATH):
        print(f"JSON 파일을 찾을 수 없습니다: {JSON_FILE_PATH}")
        return

    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = data.get("documents", [])
    for doc in documents:
        doc_id = doc.get("document_id", "")
        
        chapters = doc.get("chapters", [])
        for chapter in chapters:
            chapter_number = chapter.get("chapter_number", "unknown")
            sections = chapter.get("sections", [])
            for section in sections:
                section_number = section.get("section_number", "default")
                articles = section.get("articles", [])
                for article in articles:
                    article_number = article.get("article_number", "")
                    article_id = f"doc_{doc_id}_chap_{chapter_number}_sec_{section_number}_art_{article_number}"
                    # 조 단위 콘텐츠: 조 제목, 조 본문, 그리고 모든 단락 및 항목 텍스트를 포함
                    article_content = get_skeleton_text(article_id)
                    article_embedding = generate_embedding(article_content, model=model)
                    collection.add(
                        ids=[article_id],
                        embeddings=[article_embedding],
                        documents=[article_content]
                    )
                    print(f"조 레벨 {article_id} 삽입 완료.")
    print("모든 레벨의 임베딩 삽입 완료.")

def query_document(prompt: str, n_results: int = 1,
                   embedding_model: str = "mxbai-embed-large",
                   generation_model: str = "exaone3.5:32b") -> dict:
    """
    사용자 쿼리에 대해, 임베딩-콘텐츠 pair 중 유사도 검색을 통해 관련 레코드를 찾고,
    해당 레코드를 context로 하여 RAG 프롬프트를 구성한 후 답변을 생성합니다.
    
    Args:
        prompt (str): 사용자 입력 쿼리.
        n_results (int): 검색할 레코드 개수 (기본값 1).
        embedding_model (str): 올라마 임베딩 모델.
        generation_model (str): 올라마 생성 모델.
        
    Returns:
        dict: {
                  "retrieved_document": 검색된 레코드의 콘텐츠,
                  "generated_response": 생성된 답변
              }
    """
    # 쿼리 임베딩 생성
    embed_response = ollama.embeddings(model=embedding_model, prompt=prompt)
    query_embedding = embed_response["embedding"]

    # DB에서 유사한 레코드 검색
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    if results["documents"] and results["documents"][0]:
        documents = results["documents"][0]  # 상위 n_result 개 문서


    
    inputs = tokenizer(
    [f"{prompt} [SEP] {doc}" for doc in documents],
    return_tensors="pt",
    padding=True,
    truncation=True)
    # 모델을 사용하여 관련성 점수 계산
    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits.squeeze()

    scores = scores.tolist()
    # 점수와 문서를 함께 정렬
    ranked_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)

# 가장 높은 점수를 받은 문서 선택

    retrieved_docs = " ".join([doc for score, doc in ranked_docs])
    # 검색된 콘텐츠를 기반으로 RAG 프롬프트 구성 후 답변 생성
    rag_prompt = f"Using this data: {retrieved_docs}. Respond in Korean to this prompt: {prompt}"
    gen_response = ollama.generate(model=generation_model, prompt=rag_prompt)
    
    return {
        "retrieved_document": retrieved_docs,
        "generated_response": gen_response["response"]
    }

if __name__ == "__main__":
    # 처음 실행 시, JSON 파일로부터 모든 레벨의 임베딩을 DB에 삽입합니다.
    ingest_documents()    
    embeding_model = "mxbai-embed-large"
    generate_model = "exaone3.5:32b"
    # 예제 쿼리 실행
    user_prompt = "정보화업무 절차 알려줘."
    result = query_document(user_prompt, n_results=10,embedding_model=embeding_model, generation_model=generate_model)
    print("Retrieved document:", result["retrieved_document"])
    print("Generated response:", result["generated_response"])
