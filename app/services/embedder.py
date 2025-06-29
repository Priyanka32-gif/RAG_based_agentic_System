# function for embedding sentences
def embed_chunks(chunks, model):
    """
    Takes list of Document or str chunks, returns their embeddings.
    """

    # If chunk is a Document object, extract page_content
    if hasattr(chunks[0], "page_content"):
        sentences = [doc.page_content for doc in chunks]
    else:
        sentences = chunks

    embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)

    return embeddings
