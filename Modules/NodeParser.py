from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from pydantic import Field

# This class extends `SemanticSplitterNodeParser` and makes sure that the final chunks are not larger than `Settings.chunk_size`
# This is necessary as the original `SemanticSplitterNodeParser` may produce chunks which are too large and do not fit 
# in the window size of the llm (for path extraction) and embedding model (for embedding generation)   
class CustomSemanticSplitterNodeParser(SemanticSplitterNodeParser):
    final_sentence_splitter: SentenceSplitter = Field(init=False)
    def __init__(self, buffer_size,breakpoint_percentile_threshold, embed_model, sentence_chunk_size, sentence_chunk_overlap):
        super().__init__(
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            embed_model=embed_model,
        )
        
        self.final_sentence_splitter = SentenceSplitter(
            chunk_size=sentence_chunk_size, 
            chunk_overlap=sentence_chunk_overlap
        )

    def build_semantic_nodes_from_documents(self, documents, show_progress=False):
        # First, split documents into semantic nodes
        semantic_nodes = super().build_semantic_nodes_from_documents(documents, show_progress=show_progress)
        
        # Further split semantic nodes into smaller chunks if needed
        final_nodes = []
        for node in semantic_nodes:
            if len(node.text) > self.final_sentence_splitter.chunk_size:
                smaller_chunks = self.final_sentence_splitter.get_nodes_from_documents([node])
                final_nodes.extend(smaller_chunks)
            else:
                final_nodes.append(node)
        
        return final_nodes