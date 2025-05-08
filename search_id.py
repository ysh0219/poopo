import os
import re
import json

JSON_FILE_PATH = os.path.join("app", "data", "tech_regulations.json")

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


if __name__ == "__main__":
    # 테스트 케이스들
    test_ids = {
        "문서 레벨": "doc_6",  # 문서 정보만 출력
        "장 레벨": "doc_6_chap_2장",  # 문서+장
        "절 레벨": "doc_6_chap_2장_sec_8절",  # 문서+장+절
        "조 레벨 (예: 80조) - 완전한 ID": "doc_6_chap_2장_sec_8절_art_80조",  # 문서+장+절+조
        "조 레벨 (예: 80조) - 장, 절 생략": "doc_6_art_80조",  # 문서+조만 제공
        "조 레벨 (예: 2조의2)": "doc_1_art_56조의2",  # 조 번호에 '의'가 포함된 예시
        "없는 ID": "doc_999"  # 존재하지 않는 ID
    }
    
    for level, test_id in test_ids.items():
        print(f"\n==== {level} ({test_id}) ====")
        skeleton = get_skeleton_text(test_id)
        print(skeleton)
