"""
基于EvoAgentX的医疗RAG引擎
EvoAgentX-based Medical RAG Engine
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from evoagentx.core.logging import logger
from evoagentx.storages.base import StorageHandler
from evoagentx.rag.rag import RAGEngine
from evoagentx.rag.schema import Query, Corpus, Chunk, ChunkMetadata

from evoagentx_medical_config import (
    DATA_DIR, CACHE_DIR, STORAGE_CONFIG, RAG_CONFIG,
    PROJECT_ROOT, OUTPUTS_DIR
)

class MedicalRAGEngine:
    """基于EvoAgentX的医疗文档RAG引擎"""
    
    def __init__(self):
        """初始化医疗RAG引擎"""
        self.logger = logging.getLogger(__name__)
        
        # 初始化存储处理器
        self.storage_handler = StorageHandler(storageConfig=STORAGE_CONFIG)
        
        # 初始化RAG引擎
        self.rag_engine = RAGEngine(
            config=RAG_CONFIG,
            storage_handler=self.storage_handler
        )
        
        self.corpus_name = "medical_cases"
        self.is_indexed = False
        
        self.logger.info("医疗RAG引擎初始化完成")
    
    def index_medical_documents(self, force_reindex: bool = False) -> bool:
        """索引医学PDF文档"""
        try:
            # 检查是否已经建立索引
            if not force_reindex and self._check_existing_index():
                self.logger.info("发现现有索引，跳过重建")
                self.is_indexed = True
                return True
            
            # 获取PDF文件列表
            pdf_files = list(DATA_DIR.glob("*.pdf"))
            if not pdf_files:
                self.logger.warning(f"在 {DATA_DIR} 中未找到PDF文件")
                return False
            
            self.logger.info(f"开始索引 {len(pdf_files)} 个医学PDF文档...")
            
            # 创建语料库
            chunks = []
            
            # 添加文档到语料库
            for pdf_file in pdf_files:
                self.logger.info(f"处理文档: {pdf_file.name}")
                
                # 使用EvoAgentX的文档处理功能
                file_chunks = self._process_pdf_file(pdf_file)
                chunks.extend(file_chunks)
            
            # 创建完整的语料库
            corpus = Corpus(chunks=chunks, corpus_id=self.corpus_name)
            
            # 将语料库添加到RAG引擎并建立索引
            self.rag_engine.add(index_type="vector", nodes=corpus, corpus_id=self.corpus_name)
            
            self.is_indexed = True
            self.logger.info(f"成功索引 {len(chunks)} 个文档块")
            
            return True
            
        except Exception as e:
            self.logger.error(f"索引文档时发生错误: {str(e)}")
            return False
    
    def _process_pdf_file(self, pdf_file: Path) -> List[Chunk]:
        """处理单个PDF文件"""
        import PyPDF2
        
        chunks = []
        try:
            # 提取PDF文本
            with open(pdf_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            
            if not text.strip():
                self.logger.warning(f"无法从 {pdf_file.name} 提取文本")
                return chunks
            
            # 使用RAG配置的分块策略
            chunk_size = RAG_CONFIG.chunker.chunk_size
            chunk_overlap = RAG_CONFIG.chunker.chunk_overlap
            
            # 简单分块（可以使用更复杂的分块策略）
            text_chunks = self._chunk_text(text, chunk_size, chunk_overlap)
            
            # 创建Chunk对象
            for i, chunk_text in enumerate(text_chunks):
                if chunk_text.strip():
                    chunk = Chunk(
                        chunk_id=f"{pdf_file.stem}_{i}",
                        text=chunk_text,
                        metadata=ChunkMetadata(
                            doc_id=pdf_file.stem,
                            corpus_id=self.corpus_name,
                            chunk_index=i,
                            file_name=pdf_file.name,
                            file_path=str(pdf_file)
                        ),
                        start_char_idx=0,
                        end_char_idx=len(chunk_text),
                        excluded_embed_metadata_keys=[],
                        excluded_llm_metadata_keys=[],
                        relationships={}
                    )
                    chunks.append(chunk)
            
        except Exception as e:
            self.logger.error(f"处理PDF文件 {pdf_file} 时出错: {str(e)}")
        
        return chunks
    
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """文本分块"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # 尝试在句子边界分块
            chunk = text[start:end]
            last_sentence_end = max(
                chunk.rfind('。'), 
                chunk.rfind('!'), 
                chunk.rfind('？'),
                chunk.rfind('.'),
                chunk.rfind('!'),
                chunk.rfind('?')
            )
            
            if last_sentence_end > chunk_size // 2:
                end = start + last_sentence_end + 1
            
            chunks.append(text[start:end])
            start = end - overlap
        
        return chunks
    
    def _check_existing_index(self) -> bool:
        """检查是否存在现有索引"""
        try:
            # 检查存储目录是否存在相关文件
            index_path = Path(STORAGE_CONFIG.path)
            if index_path.exists() and any(index_path.iterdir()):
                return True
            
            # 可以添加更详细的索引存在性检查
            return False
            
        except Exception:
            return False
    
    def search_similar_cases(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """搜索相似医学病例"""
        try:
            if not self.is_indexed:
                self.logger.error("RAG引擎尚未建立索引")
                return {
                    "status": "error",
                    "error": "RAG引擎尚未建立索引",
                    "results": []
                }
            
            # 创建查询对象
            query = Query(query_str=query_text, top_k=top_k)
            
            # 执行检索
            rag_results = self.rag_engine.query(query, corpus_id=self.corpus_name)
            
            # 格式化结果
            formatted_results = []
            if rag_results.corpus and rag_results.corpus.chunks:
                for i, chunk in enumerate(rag_results.corpus.chunks[:top_k]):
                    formatted_result = {
                        "rank": i + 1,
                        "content": chunk.text,
                        "score": getattr(chunk, 'score', 1.0),  # 如果没有分数，使用默认值
                        "source": chunk.metadata.file_path,
                        "document_title": chunk.metadata.file_name,
                        "chunk_id": chunk.chunk_id,
                        "metadata": {
                            "chunk_index": chunk.metadata.chunk_index,
                            "doc_id": chunk.metadata.doc_id
                        }
                    }
                    formatted_results.append(formatted_result)
            
            return {
                "status": "success",
                "query": query_text,
                "total_results": len(formatted_results),
                "results": formatted_results,
                "search_metadata": {
                    "corpus_name": self.corpus_name,
                    "top_k": top_k,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"检索过程中发生错误: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "results": []
            }
    
    def get_corpus_info(self) -> Dict[str, Any]:
        """获取语料库信息"""
        try:
            # 这里可以添加获取语料库统计信息的逻辑
            return {
                "corpus_name": self.corpus_name,
                "is_indexed": self.is_indexed,
                "storage_path": str(STORAGE_CONFIG.path),
                "total_documents": len(list(DATA_DIR.glob("*.pdf"))),
                "rag_config": {
                    "chunk_size": RAG_CONFIG.chunker.chunk_size,
                    "chunk_overlap": RAG_CONFIG.chunker.chunk_overlap,
                    "embedding_model": RAG_CONFIG.embedding.model_name,
                    "top_k": RAG_CONFIG.retrieval.top_k
                }
            }
        except Exception as e:
            self.logger.error(f"获取语料库信息时出错: {str(e)}")
            return {}
    
    def save_search_results(self, results: Dict[str, Any], query: str) -> str:
        """保存搜索结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"medical_search_{timestamp}.json"
            filepath = OUTPUTS_DIR / "results" / filename
            
            # 确保目录存在
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存结果
            save_data = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "engine_info": self.get_corpus_info(),
                "search_results": results
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"搜索结果已保存到: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"保存搜索结果时出错: {str(e)}")
            return ""

def main():
    """测试医疗RAG引擎"""
    print("🏥 初始化医疗RAG引擎...")
    
    # 创建引擎
    medical_rag = MedicalRAGEngine()
    
    # 建立索引
    print("📚 正在索引医学文档...")
    success = medical_rag.index_medical_documents()
    
    if not success:
        print("❌ 文档索引失败")
        return
    
    print("✅ 文档索引完成")
    
    # 测试搜索
    test_queries = [
        "患者出现头痛症状",
        "血压升高伴视物模糊",
        "胸痛呼吸困难心电图异常"
    ]
    
    for query in test_queries:
        print(f"\n🔍 搜索: {query}")
        results = medical_rag.search_similar_cases(query, top_k=3)
        
        if results["status"] == "success":
            print(f"找到 {results['total_results']} 个相似病例:")
            for result in results["results"]:
                print(f"  - 排名 {result['rank']}: {result['document_title']}")
                print(f"    相似度: {result['score']:.4f}")
                print(f"    内容: {result['content'][:100]}...")
                print()
        else:
            print(f"❌ 搜索失败: {results.get('error', '未知错误')}")
    
    # 显示语料库信息
    corpus_info = medical_rag.get_corpus_info()
    print("\n📊 语料库信息:")
    print(json.dumps(corpus_info, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()