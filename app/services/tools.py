from app.services.vector_store import search_cosine, search_dot


def search_docs(input: str, embedding_model):
    """
    Parse input string to extract query and search_method, then search.
    Expects input format: "<search_method>::<query>"
    e.g. "cosine::What is LangChain?"
    """
    try:
        if "::" in input:
            search_method, query = input.split("::", 1)
            query = query.strip()
        else:
            search_method = "cosine"  # default to cosine
            query = input.strip()

        # Compute OpenAI embedding for the query
        query_vector = embedding_model.embed_query(query)

        if search_method == 'cosine':
            results = search_cosine(query_vector, top_k=5)
        elif search_method == 'dot':
            results = search_dot(query_vector, top_k=5)
        else:
            return f"Unknown search method: {search_method}"

        matched_chunks = [hit.payload.get("text", "") for hit in results]
        return "\n---\n".join(matched_chunks)

    except Exception as e:
        return f"Error during document search: {str(e)}"
