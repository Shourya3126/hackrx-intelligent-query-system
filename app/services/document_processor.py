import requests
import PyPDF2
import pdfplumber
from docx import Document
import io
from typing import List, Dict, Any
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    async def process_document_from_url(self, url: str) -> List[Dict[str, Any]]:
        """Download and process document from URL"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            file_content = response.content
            
            if 'pdf' in content_type or url.lower().endswith('.pdf'):
                text = self._extract_pdf_text(file_content)
            elif 'word' in content_type or url.lower().endswith(('.docx', '.doc')):
                text = self._extract_docx_text(file_content)
            else:
                # Try PDF first, then DOCX
                try:
                    text = self._extract_pdf_text(file_content)
                except:
                    text = self._extract_docx_text(file_content)
            
            chunks = self._create_chunks(text, url)
            logger.info(f"Processed document: {len(chunks)} chunks created")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document from {url}: {str(e)}")
            raise
    
    def _extract_pdf_text(self, file_content: bytes) -> str:
        """Extract text from PDF using pdfplumber for better accuracy"""
        text_parts = []
        
        with io.BytesIO(file_content) as pdf_file:
            with pdfplumber.open(pdf_file) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
        
        return "\n\n".join(text_parts)
    
    def _extract_docx_text(self, file_content: bytes) -> str:
        """Extract text from DOCX"""
        doc = Document(io.BytesIO(file_content))
        text_parts = []
        
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)
        
        return "\n\n".join(text_parts)
    
    def _create_chunks(self, text: str, source_url: str) -> List[Dict[str, Any]]:
        """Create overlapping text chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": source_url,
                    "chunk_index": len(chunks),
                    "word_count": len(chunk_words)
                }
            })
            
            if i + self.chunk_size >= len(words):
                break
        
        return chunks
