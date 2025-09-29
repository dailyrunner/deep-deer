"""
Example usage of different model providers in Deep Deer

This example shows how to use:
1. Ollama (default)
2. Qwen local model (lazy-loaded)
3. E5 Instruct embeddings (lazy-loaded)
"""

import asyncio
import os
from app.services.llm_provider import LLMService, ModelProvider
from app.services.embeddings import EmbeddingService, EmbeddingProvider
from app.core.config import settings


async def example_ollama():
    """Example using Ollama provider (default)"""
    print("\n=== Ollama Provider Example ===")

    llm_service = LLMService()

    # Initialize with Ollama
    await llm_service.initialize({
        "enable_ollama": True,
        "ollama_model": "gpt-oss:120b",  # or any model you have
        "ollama_base_url": "http://localhost:11434"
    })

    # Generate text
    response = await llm_service.generate(
        prompt="What is deep learning?",
        temperature=0.7,
        max_tokens=200
    )
    print(f"Ollama Response: {response[:200]}...")

    await llm_service.cleanup()


async def example_qwen():
    """Example using Qwen local model with lazy loading"""
    print("\n=== Qwen Local Model Example (Lazy Loaded) ===")

    llm_service = LLMService()

    # Initialize with Qwen - model will be loaded on first use
    await llm_service.initialize({
        "enable_qwen_local": True,
        "qwen_model": "Qwen/Qwen3-0.6B"  # Small model for testing
    })

    print("Qwen provider initialized (model not loaded yet)")

    # First generation - model will be loaded here
    print("First generation (loading model)...")
    response1 = await llm_service.generate(
        prompt="Explain AI in one sentence.",
        temperature=0.5,
        max_tokens=50
    )
    print(f"Qwen Response 1: {response1}")

    # Second generation - uses cached model
    print("Second generation (using cached model)...")
    response2 = await llm_service.generate(
        prompt="What is machine learning?",
        temperature=0.5,
        max_tokens=50
    )
    print(f"Qwen Response 2: {response2}")

    await llm_service.cleanup()


async def example_e5_embeddings():
    """Example using E5 Instruct embeddings with lazy loading"""
    print("\n=== E5 Instruct Embeddings Example (Lazy Loaded) ===")

    embedding_service = EmbeddingService()

    # Initialize with E5 Instruct
    embedding_service.initialize(
        provider_type=EmbeddingProvider.E5_INSTRUCT,
        model_name="intfloat/multilingual-e5-large-instruct"
    )

    print("E5 Instruct provider initialized (model not loaded yet)")

    # Test documents
    documents = [
        "Deep learning is a subset of machine learning.",
        "Python is a popular programming language.",
        "딥러닝은 머신러닝의 한 분야입니다.",  # Korean text
        "深度学习是机器学习的一个子集。"  # Chinese text
    ]

    # First embedding - model will be loaded here
    print("Embedding documents (loading model)...")
    doc_embeddings = embedding_service.embed_documents(documents)
    print(f"Document embeddings shape: {len(doc_embeddings)} x {len(doc_embeddings[0])}")

    # Query embedding - uses cached model
    print("Embedding query (using cached model)...")
    query = "What is deep learning?"
    query_embedding = embedding_service.embed_query(query)
    print(f"Query embedding shape: {len(query_embedding)}")

    # Calculate similarity (cosine similarity)
    import numpy as np

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print("\nSimilarity scores:")
    for i, doc in enumerate(documents):
        similarity = cosine_similarity(query_embedding, doc_embeddings[i])
        print(f"  '{doc[:50]}...' : {similarity:.4f}")

    embedding_service.cleanup()


async def example_combined():
    """Example using both Qwen for generation and E5 for embeddings"""
    print("\n=== Combined Example: Qwen + E5 ===")

    # Initialize services
    llm_service = LLMService()
    embedding_service = EmbeddingService()

    # Setup Qwen for generation
    await llm_service.initialize({
        "enable_qwen_local": True,
        "qwen_model": "Qwen/Qwen3-0.6B"
    })

    # Setup E5 for embeddings
    embedding_service.initialize(
        provider_type=EmbeddingProvider.E5_INSTRUCT,
        model_name="intfloat/multilingual-e5-large-instruct"
    )

    # Example RAG-like workflow
    documents = [
        "The capital of France is Paris. It is known for the Eiffel Tower.",
        "Python was created by Guido van Rossum in 1991.",
        "Deep learning uses neural networks with multiple layers.",
    ]

    query = "Tell me about France"

    # 1. Embed documents and query
    doc_embeddings = embedding_service.embed_documents(documents)
    query_embedding = embedding_service.embed_query(query)

    # 2. Find most relevant document
    import numpy as np
    similarities = [
        np.dot(query_embedding, doc_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
        for doc_emb in doc_embeddings
    ]
    best_doc_idx = np.argmax(similarities)
    best_doc = documents[best_doc_idx]

    print(f"Most relevant document: '{best_doc}'")
    print(f"Similarity score: {similarities[best_doc_idx]:.4f}")

    # 3. Generate response using context
    prompt = f"""Context: {best_doc}

Question: {query}
Answer:"""

    response = await llm_service.generate(
        prompt=prompt,
        temperature=0.7,
        max_tokens=100
    )

    print(f"\nGenerated Response: {response}")

    # Cleanup
    await llm_service.cleanup()
    embedding_service.cleanup()


async def main():
    """Run all examples"""

    # Check which models are available
    print("=" * 60)
    print("Deep Deer Model Provider Examples")
    print("=" * 60)

    # You can set these in environment or .env file
    # os.environ["ENABLE_QWEN_LOCAL"] = "true"
    # os.environ["ENABLE_E5_INSTRUCT"] = "true"

    try:
        # Run Ollama example if Ollama is running
        try:
            await example_ollama()
        except Exception as e:
            print(f"Ollama example failed: {e}")
            print("Make sure Ollama is running locally")

        # Run Qwen example (will download model on first run)
        await example_qwen()

        # Run E5 embeddings example (will download model on first run)
        await example_e5_embeddings()

        # Run combined example
        await example_combined()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())