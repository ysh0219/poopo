import ollama
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings

# 같은 "chroma_db" 폴더를 사용하는 Persistent DB에 연결
client = chromadb.PersistentClient(
    path="chroma_db",
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

# 저장된 컬렉션 이름("docs")으로 컬렉션 가져오기
collection = client.get_collection(name="docs")


# 전체 레코드 정보를 가져옵니다.
records = collection.get()
all_ids = records.get("ids", [])

# 각 레벨 별 필터링
doc_ids = [record_id for record_id in all_ids if record_id.startswith("doc_") and "chap_" not in record_id]
chapter_ids = [record_id for record_id in all_ids if "chap_" in record_id and "art_" not in record_id]
article_ids = [record_id for record_id in all_ids if "art_" in record_id]

print("전체 레코드 개수:", len(all_ids))
print("문서(document) 레벨 임베딩 개수:", len(doc_ids))
print("장(chapter) 레벨 임베딩 개수:", len(chapter_ids))
print("조(article) 레벨 임베딩 개수:", len(article_ids))



# 특정 ID의 레코드 조회
record = collection.get(ids=["doc_1"])
print("doc_1 내용:")
print(record.get("documents", []))

record = collection.get(ids=["doc_1_chap_1장"])
print("doc_1_chap_1장 내용:")
print(record.get("documents", []))

record = collection.get(ids=["doc_1_chap_1장_sec_default_art_1조"])
print("doc_1_chap_1장_sec_default_art_1조 내용:")
print(record.get("documents", []))

