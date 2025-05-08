import argparse
import json
import os
import ollama
import chromadb
import re
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 모델과 토크나이저 로드
model_name = "monologg/koelectra-base-v3-discriminator"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# DB 및 데이터 파일 경로 설정
JSON_FILE_PATH = os.path.join("app", "data", "tech_regulations.json")

def get_skeleton_text(record_id: str, json_file_path: str = JSON_FILE_PATH) -> str:
    # ... (생략: 기존 코드와 동일)
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return f"JSON 파일을 로드하는 데 실패했습니다: {e}"
        
    documents = data.get("documents", [])
    pattern = r"doc_(?P<doc_id>[^_]+)(?:_chap_(?P<chap>[^_]+))?(?:_sec_(?P<sec>[^_]+))?(?:_art_(?P<art>.+))?"
    m = re.match(pattern, record_id)
    if not m:
        return f"올바르지 않은 ID 형식: {record_id}"
    
    groups = m.groupdict()
    doc_id = groups.get("doc_id")
    chap = groups.get("chap")
    sec = groups.get("sec")
    art = groups.get("art")
    
    target_doc = None
    for doc in documents:
        if str(doc.get("document_id", "")) == doc_id:
            target_doc = doc
            break
    if target_doc is None:
        return f"문서(document) ID '{doc_id}'를 찾지 못했습니다."
    
    lines = []
    doc_title = target_doc.get("document_title", "")
    promulgation = target_doc.get("promulgation_number", "")
    lines.append(doc_title)
    lines.append(promulgation)
    
    article_found = None
    found_chapter = None
    found_section = None

    if art is not None:
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

    if art is None:
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
    
    if article_found is None:
        return "\n".join(lines) + f"\n조(article) 번호 '{art}'을(를) 찾지 못했습니다."
    
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
    
    art_num = article_found.get("article_number", "")
    art_title = article_found.get("article_title", "")
    art_text = article_found.get("article_text", "").strip()
    if not art_num.startswith("제"):
        art_num = "제" + art_num
    article_line = f"      ├─ {art_num}({art_title}): {art_text}".rstrip()
    lines.append(article_line)
    
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
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

def ingest_documents(model: str = "mxbai-embed-large"):
    # 모델별 DB 경로 지정
    db_path = f"chroma_db_{model}"
    
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    
    # 컬렉션이 이미 존재하면 건너뛰도록 처리 (단순하게 client의 컬렉션 리스트 길이로 판단)
    existing_collections = client.list_collections()
    if len(existing_collections) > 0:
        print(f"DB 경로 '{db_path}'에 이미 컬렉션이 존재합니다. Ingestion을 건너뜁니다.")
        return

    # 컬렉션이 없으면 새로 생성
    collection = client.create_collection(name="docs")
    
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
                    article_content = get_skeleton_text(article_id)
                    article_embedding = generate_embedding(article_content, model=model)
                    collection.add(
                        ids=[article_id],
                        embeddings=[article_embedding],
                        documents=[article_content]
                    )
                    print(f"조 레벨 {article_id} 삽입 완료.")
    print("모든 레벨의 임베딩 삽입 완료.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="임베딩 기법 이름을 인자로 받아 문서 임베딩 삽입 실행")
    parser.add_argument("--embedding", type=str, required=True, help="사용할 임베딩 기법 이름 (예: mxbai-embed-large)")
    args = parser.parse_args()

    # 입력받은 임베딩 기법 이름에 따라 ingest_documents 실행
    ingest_documents(model=args.embedding)
