"""
PDF文档导入模块 - 将医学案例报告导入FAISS向量数据库
PDF Document Ingestion Module - Import medical case reports into FAISS vector database
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# 添加EvoAgentX到路径
sys.path.append(str(Path(__file__).parent.parent.parent / "EvoAgentX-clean_tools"))

from config import FAISS_CONFIG, LLM_CONFIG, DATA_DIR, load_api_key, ensure_directories

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalPDFProcessor:
    """医学PDF文档处理器"""
    
    def __init__(self):
        ensure_directories()
        
        # 初始化OpenAI嵌入模型
        api_key = load_api_key(LLM_CONFIG["api_key_file"])
        self.embeddings = OpenAIEmbeddings(
            model=FAISS_CONFIG["embedding_model"],
            openai_api_key=api_key
        )
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=FAISS_CONFIG["chunk_size"],
            chunk_overlap=FAISS_CONFIG["chunk_overlap"],
            separators=["\n\n", "\n", "。", "；", "！", "？", " ", ""]
        )
        
        self.vector_store = None
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """从PDF文件提取文本"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"提取PDF文本失败 {pdf_path}: {str(e)}")
            return ""
    
    def process_pdf_files(self, pdf_directory: str = None) -> List[Document]:
        """处理PDF文件并创建文档对象"""
        if pdf_directory is None:
            pdf_directory = DATA_DIR
        
        documents = []
        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        
        logger.info(f"发现 {len(pdf_files)} 个PDF文件待处理")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            logger.info(f"正在处理: {pdf_file}")
            
            # 提取文本
            text = self.extract_text_from_pdf(pdf_path)
            if not text.strip():
                logger.warning(f"PDF文件 {pdf_file} 无法提取文本")
                continue
            
            # 分割文本
            chunks = self.text_splitter.split_text(text)
            
            # 创建文档对象
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # 确保块不为空
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": pdf_file,
                            "chunk_id": i,
                            "file_path": pdf_path,
                            "doc_type": "medical_case_report",
                            "corpus_id": FAISS_CONFIG["corpus_id"]
                        }
                    )
                    documents.append(doc)
        
        logger.info(f"总共创建了 {len(documents)} 个文档块")
        return documents
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """创建FAISS向量存储"""
        if not documents:
            raise ValueError("文档列表为空，无法创建向量存储")
        
        logger.info("正在创建FAISS向量存储...")
        vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # 保存向量存储
        index_path = FAISS_CONFIG["index_path"]
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        vector_store.save_local(index_path)
        logger.info(f"向量存储已保存到: {index_path}")
        
        return vector_store
    
    def load_vector_store(self) -> FAISS:
        """加载现有的向量存储"""
        index_path = FAISS_CONFIG["index_path"]
        if os.path.exists(index_path):
            logger.info(f"正在加载现有向量存储: {index_path}")
            return FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            raise FileNotFoundError(f"向量存储不存在: {index_path}")
    
    def search_similar_cases(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """搜索相似病例"""
        if k is None:
            k = FAISS_CONFIG["top_k"]
        
        if self.vector_store is None:
            try:
                self.vector_store = self.load_vector_store()
            except FileNotFoundError:
                logger.error("向量存储不存在，请先运行文档导入")
                return []
        
        # 执行相似性搜索
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        # 格式化结果
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(score),
                "source": doc.metadata.get("source", "unknown")
            })
        
        return formatted_results

def main():
    """主函数 - 执行PDF导入流程"""
    processor = MedicalPDFProcessor()
    
    try:
        # 处理PDF文件
        documents = processor.process_pdf_files()
        
        if not documents:
            logger.error("没有成功处理任何PDF文件")
            return
        
        # 创建向量存储
        vector_store = processor.create_vector_store(documents)
        processor.vector_store = vector_store
        
        # 测试搜索功能
        test_query = "患者出现头痛症状"
        logger.info(f"测试搜索: {test_query}")
        results = processor.search_similar_cases(test_query, k=3)
        
        for i, result in enumerate(results, 1):
            logger.info(f"结果 {i}:")
            logger.info(f"  相似度分数: {result['similarity_score']}")
            logger.info(f"  来源: {result['source']}")
            logger.info(f"  内容片段: {result['content'][:200]}...")
            logger.info("---")
        
        logger.info("PDF导入和索引创建完成！")
        
    except Exception as e:
        logger.error(f"处理过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()