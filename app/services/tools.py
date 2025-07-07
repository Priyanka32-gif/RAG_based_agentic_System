from app.services.vector_store import search_cosine, search_dot
import logging
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
            start_time = time.perf_counter()
            results = search_cosine(query_vector, top_k=5)
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(f"latency {latency_ms}")
            # Print detailed cosine info
            logger.debug(f"{'RANK':<6} {'COSINE_ID':<36} {'COSINE_SCORE':<15}")
            logger.debug("-" * 20)
            for i, point in enumerate(results[:5]):
                logger.debug(f"{i + 1:<6} {str(point.id):<36} {point.score:<15.4f}")
                logger.debug(f"   TEXT: {point.payload.get('text', '')[:80]}...")
            # Return matched chunks text

        elif search_method == 'dot':
            start_time = time.perf_counter()
            results = search_dot(query_vector, top_k=5)
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(f"latency {latency_ms}")
            logger.debug(f"{'RANK':<6} {'DOT_ID':<36} {'DOT_SCORE':<15}")
            logger.debug("-" * 20)
            for i, point in enumerate(results[:5]):
                logger.debug(f"{i + 1:<6} {str(point.id):<36} {point.score:<15.4f}")
                logger.debug(f"   TEXT: {point.payload.get('text', '')[:80]}...")

        else:
            return f"Unknown search method: {search_method}"

        matched_chunks = [hit.payload.get("text", "") for hit in results]
        print(matched_chunks)
        return "\n---\n".join(matched_chunks)

    except Exception as e:
        return f"Error during document search: {str(e)}"

