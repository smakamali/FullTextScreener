import numpy as np

def find_second_occurrence(s, substring):
    first_occurrence = s.find(substring)
    if first_occurrence == -1:
        return -1
    second_occurrence = s.find(substring, first_occurrence + 1)
    return second_occurrence

def get_first_split(triplet_string):
    # Split the string based on the common entity
    split_point = find_second_occurrence(triplet_string, '[relationship:{relationship')
    if split_point > 0:
        first_split_end = triplet_string.rfind('}', 0, split_point)+1
        first_triplet = triplet_string[:first_split_end]
        return first_triplet
    else:
        return triplet_string

def reverse_triplet(triplet):
    # Split the triplet into parts based on the relationship arrows
    if '->' in triplet:
        parts = triplet.split(']-> ')
        object_predicate = parts[0].split(' -[')
        obj = object_predicate[0].strip()
        predicate = object_predicate[1].strip()
        subject = parts[1].strip()
    
    elif '<-' in triplet:
        parts = triplet.split(' <-[')
        subject = parts[0].strip()
        predicate_object = parts[1].split(']- ', 1)
        predicate = predicate_object[0].strip()
        obj = predicate_object[1].strip()
        
    else:
        raise ValueError("Invalid triplet format")
    
    # Determine the direction of the relationship
    if '->' in triplet:
        reversed_triplet = f"{subject} <-[{predicate}]- {obj}"
    else:
        reversed_triplet = f"{obj} -[{predicate}]-> {subject}"
    
    return reversed_triplet


# a function that takes the AgentChatResponse as input and 
#   returns citations
def get_kg_agent_citations(node, kg_index):
    meta_data_text = node.metadata['kg_rel_text']
    citations = []
    for input_string in meta_data_text:
        first_triplet = get_first_split(input_string)
        search_node_id = kg_index._index_struct.search_node_by_keyword(first_triplet)
        search_node_id.extend(kg_index._index_struct.search_node_by_keyword(reverse_triplet(first_triplet)))
        nodes = kg_index.docstore.get_nodes(search_node_id)
        if len(nodes) == 0:
            print("Triplet not found in index structure!")
        else:
            citations.append(nodes[0].metadata['citation'])
    return citations

def get_agent_citations(AgentChatResponse,kg_index):
    citations = []
    for node in AgentChatResponse.source_nodes:
        if 'kg_rel_text' in node.metadata.keys():
            citations.extend(get_kg_agent_citations(node, kg_index))
        if 'citation' in node.metadata.keys():
            citations.append(node.metadata['citation'])
    citations = list(np.unique(np.array(citations)))
    return citations

def getAgentCitationsN4j(resp,cutoff_score=0.9):
    
    data = []
    dtype = [('key', 'U400'), ('value', 'f4')]
    for node in resp.source_nodes:
        if 'citation' in node.metadata.keys():
            data.append((node.metadata['citation'],float(node.score)))
    data = np.array(data, dtype=dtype)

    # Sort the array by the 'key' field
    data.sort(order='key')

    # Use np.unique to find the unique keys and the starting index for each group
    unique_keys, indices = np.unique(data['key'], return_index=True)

    # Calculate the mean for each group, including the last group
    group_means = [data['value'][indices[i]: indices[i + 1] if i + 1 < len(indices) else None].max() for i in range(len(indices))]

    # Combine the unique keys with their aggregated means
    result = np.array(list(zip(unique_keys, group_means)), dtype=dtype)
    result=np.sort(result,order='value')[::-1]

    # Eliminate references with zero score
    result=result[result['value'] >= cutoff_score]
    
    return result

def formatReferences(citations):
    references=''
    if citations.shape[0]>0:
        references = '<br><p style="font-size:12px;">**The following references were reviewed to provide the above answer:**<br><ol type="1" style="font-size:12px;">'
        for _,cit in enumerate(citations):
            references+= '<li> {}  <i><b>Relevance Score: </b>{}</i></li>'.format(cit['key'],str(cit['value']))
        references+='</ol></p>'
    return references