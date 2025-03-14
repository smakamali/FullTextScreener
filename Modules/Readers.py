import pandas as pd
from llama_index.core.readers.base import BaseReader
from llama_index.core import Document
from pathlib import Path
from typing import Dict, List, Optional, Union
import io

class CustomCSVReader(BaseReader):
    def __init__(self):
        super().__init__()
        
    def load_data(
        self,
        file: Union[Path, str],
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        
        if not isinstance(file, Path):
            file = Path(file)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file,header=0,dtype=str)
        
        # Initialize an empty list to store documents
        documents = []
        
        # Iterate over each row in the DataFrame
        for _, row in df.iterrows():
            # Create a document for each row
            text = row.Abstract
            citation = ', '.join([row.Title, row.Source, row.URL])
            metadata = {
                "title": row.Title,
                "source": row.Source,
                "source_url": row.URL,
                "citation": citation,
            }
            
            if extra_info is not None:
                metadata.update(extra_info)
            
            document = Document(text=text, metadata=metadata)
            documents.append(document)
        
        return documents

from llama_index.readers.file.docs.base import PDFReader

class CustomPDFReader(PDFReader):
    def __init__(self, llm):
        self._llm = llm
        super().__init__()
        
    def load_data(
        self,
        file: Union[Path, str],
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        
        if not isinstance(file, Path):
            file = Path(file)
        
        try:
            import pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber is required to read PDF files: `pip install pdfplumber`"
            )
        
        # Open the PDF file
        pdf = pdfplumber.open(file)
            
        # Get the number of pages in the PDF document
        num_pages = len(pdf.pages)

        docs = []

        # This block returns a whole PDF as a single Document
        if self.return_full_document:
            metadata = {"file_name": file.name}
            if extra_info is not None:
                metadata.update(extra_info)

            # Join text extracted from each page
            text = "\n".join(
                pdf.pages[page].extract_text() for page in range(num_pages)
            )

            docs.append(Document(text=text, metadata=metadata))

        # This block returns each page of a PDF as its own Document
        else:
            # Iterate over every page

            for page in range(num_pages):
                # Extract the text from the page
                page_text = pdf.pages[page].extract_text()
                page_label = page

                # extract the citation from the first page of the paper using LLM
                if page == 0:
                    prompt_str = 'Extract the paper citation from the information provided below based on this format:\n ``` \n author name(s) (last name, first name;), paper title, publication, year, page number, doi\n ```\n paper\'s first page content:\n{}\n ATTENTION: RETURN THE PAPER CITATION ONLY, DO NOT RETURN ANY OTHER INFORMATION, DO NOT REPEAT:'.format(page_text)
                    citation = self._llm.complete(prompt_str).text
                    print("=====> Citation:\n", citation)

                metadata = {"page_label": page_label, "file_name": file.name, "citation": citation}
                if extra_info is not None:
                    metadata.update(extra_info)

                docs.append(Document(text=page_text, metadata=metadata))

        return docs
