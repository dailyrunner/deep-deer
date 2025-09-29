"""
Test script for HuggingFace Local model integration including Qwen
"""
import asyncio
import os
from app.services.llm_provider import LLMService
from app.services.langchain_pipeline import LangChainPipeline
from app.core.config import settings


async def test_huggingface_with_qwen():
    """Test HuggingFace Local provider with Qwen model"""
    print("\n" + "="*60)
    print("Testing HuggingFace Local Provider with Qwen Model")
    print("="*60)

    # Create LLM service
    llm_service = LLMService()

    # Test 1: Initialize with HuggingFace Local (Qwen model)
    print("\n1. Initializing HuggingFace Local provider with Qwen...")
    config = {
        "enable_huggingface_local": True,
        "huggingface_local_model": "Qwen/Qwen3-0.6B",
        "enable_ollama": False  # Disable Ollama to test HuggingFace only
    }

    success = await llm_service.initialize(config)
    print(f"   Initialization: {'✓ Success' if success else '✗ Failed'}")
    print(f"   Available providers: {llm_service.list_providers()}")
    print(f"   Default provider: {llm_service.get_default_provider()}")

    if not success:
        print("   Failed to initialize HuggingFace provider")
        return

    # Test 2: Direct generation
    print("\n2. Testing direct generation...")
    try:
        prompt = "What is artificial intelligence in one sentence?"
        response = await llm_service.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=50
        )
        print(f"   Prompt: {prompt}")
        print(f"   Response: {response[:200]}...")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 3: LangChain compatibility
    print("\n3. Testing LangChain compatibility...")
    try:
        provider = llm_service.providers.get("huggingface_local")
        if provider:
            print(f"   LangChain LLM type: {type(provider.llm)}")
            print(f"   Has invoke method: {hasattr(provider.llm, 'invoke')}")

            # Test invoke
            test_response = provider.llm.invoke("Hello, how are you?")
            print(f"   Test invoke response: {test_response[:100]}...")
    except Exception as e:
        print(f"   ✗ Error testing LangChain compatibility: {e}")

    # Test 4: Chain functionality
    print("\n4. Testing LangChain chain...")
    try:
        template = "Translate the following to {language}: {text}"
        response = await llm_service.generate_with_chain(
            prompt_template=template,
            input_variables={
                "language": "French",
                "text": "Hello world"
            }
        )
        print(f"   Chain response: {response}")
    except Exception as e:
        print(f"   ✗ Error in chain: {e}")

    await llm_service.cleanup()


async def test_huggingface_with_other_models():
    """Test HuggingFace Local provider with different models"""
    print("\n" + "="*60)
    print("Testing HuggingFace Local with Different Models")
    print("="*60)

    test_models = [
        "gpt2",  # Small English model
        # "microsoft/phi-2",  # Uncomment if you want to test
        # "meta-llama/Llama-2-7b-hf",  # Requires HF token
    ]

    for model_name in test_models:
        print(f"\n--- Testing {model_name} ---")

        llm_service = LLMService()
        config = {
            "enable_huggingface_local": True,
            "huggingface_local_model": model_name,
            "enable_ollama": False
        }

        try:
            await llm_service.initialize(config)
            response = await llm_service.generate(
                "Write a haiku about AI",
                temperature=0.8,
                max_tokens=50
            )
            print(f"   Response: {response[:150]}...")
        except Exception as e:
            print(f"   ✗ Error: {e}")
        finally:
            await llm_service.cleanup()


async def test_pipeline_integration():
    """Test integration with LangChain pipeline"""
    print("\n" + "="*60)
    print("Testing LangChain Pipeline Integration")
    print("="*60)

    # Initialize LLM service with HuggingFace Local
    llm_service = LLMService()
    config = {
        "enable_huggingface_local": True,
        "huggingface_local_model": "Qwen/Qwen3-0.6B",
        "enable_ollama": False
    }

    await llm_service.initialize(config)

    # Initialize pipeline (should use HuggingFace from llm_service)
    pipeline = LangChainPipeline()
    await pipeline.initialize()

    print(f"\n✓ Pipeline initialized")
    print(f"  LLM type: {type(pipeline.llm)}")
    print(f"  Pipeline ready: {pipeline.is_ready}")

    if pipeline.is_ready:
        # Test a simple question
        try:
            print("\nTesting pipeline question answering...")
            answer = await pipeline.answer_question(
                "What is Python?",
                store_names=None
            )
            print(f"  Answer: {answer[:200]}...")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    await llm_service.cleanup()


async def test_provider_switching():
    """Test switching between Ollama and HuggingFace providers"""
    print("\n" + "="*60)
    print("Testing Provider Switching")
    print("="*60)

    llm_service = LLMService()

    # Test 1: Start with Ollama
    print("\n1. Initializing with Ollama...")
    config = {
        "enable_ollama": True,
        "enable_huggingface_local": False
    }
    await llm_service.initialize(config)
    print(f"   Default provider: {llm_service.get_default_provider()}")

    # Cleanup and switch
    await llm_service.cleanup()

    # Test 2: Switch to HuggingFace
    print("\n2. Switching to HuggingFace Local...")
    config = {
        "enable_ollama": False,
        "enable_huggingface_local": True,
        "huggingface_local_model": "Qwen/Qwen3-0.6B"
    }
    await llm_service.initialize(config)
    print(f"   Default provider: {llm_service.get_default_provider()}")

    await llm_service.cleanup()


async def main():
    """Run all tests"""
    print("\n" + "#"*60)
    print("# HuggingFace Local Provider Test Suite")
    print("#"*60)

    # Set environment for testing
    os.environ["ENABLE_HUGGINGFACE_LOCAL"] = "true"
    os.environ["HUGGINGFACE_LOCAL_MODEL"] = "Qwen/Qwen3-0.6B"

    try:
        # Test 1: HuggingFace with Qwen
        await test_huggingface_with_qwen()

        # Test 2: Different models (optional)
        # await test_huggingface_with_other_models()

        # Test 3: Pipeline integration
        await test_pipeline_integration()

        # Test 4: Provider switching
        await test_provider_switching()

        print("\n" + "#"*60)
        print("# All tests completed!")
        print("#"*60)

    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())