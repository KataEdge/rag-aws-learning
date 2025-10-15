from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

class DocumentLoader:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def load_pdf(self, file_path):
        """PDFファイルを読み込み"""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def load_text(self, file_path):
        """テキストファイルを読み込み"""
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def load_directory(self, directory_path):
        """ディレクトリ内の全ファイルを読み込み"""
        all_documents = []
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if filename.endswith('.pdf'):
                all_documents.extend(self.load_pdf(file_path))
            elif filename.endswith('.txt'):
                all_documents.extend(self.load_text(file_path))
        return all_documents