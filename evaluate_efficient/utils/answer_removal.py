"""
Answer Removal Utilities

Provides LLM-based and regex-based methods to remove answers from reasoning traces.
"""

import os
import re
import logging
import asyncio
import aiohttp
from typing import List, Dict, Any
from tqdm.asyncio import tqdm as async_tqdm

logger = logging.getLogger(__name__)


async def remove_answer_with_gpt4o_mini(teacher_thinking: str, teacher_answer: str,
                                       api_key: str, semaphore: asyncio.Semaphore) -> str:
    """
    Use GPT-4o-mini to intelligently remove the answer from teacher thinking.

    Args:
        teacher_thinking: The full thinking trace including the answer
        teacher_answer: The answer to be removed
        api_key: OpenAI API key
        semaphore: Asyncio semaphore to limit concurrent requests

    Returns:
        The thinking trace with the answer removed
    """
    prompt = f"""You are tasked with removing the answer from a reasoning trace while preserving the thinking steps that do not reveal the answer.

Here is the reasoning trace:
{teacher_thinking}

Here is the answer that needs to be removed:
{teacher_answer}

IMPORTANT: Remove ALL statements that contain or reveal the answer, including:
1. Direct statements of the answer (e.g., "The answer is X")
2. Descriptions that include the answer content (e.g., "Elements in spiral order: X Y Z")
3. Any intermediate steps that state the final result
4. Any reformulations or paraphrases of the answer

Keep ONLY the reasoning steps and intermediate work that do NOT reveal what the final answer is.

Return ONLY the cleaned reasoning trace with no answer content, with no additional commentary or explanation."""

    async with semaphore:
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }

                payload = {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 2048
                }

                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"].strip()
                    else:
                        error_text = await response.text()
                        logger.error(f"API error {response.status}: {error_text}")
                        # Fallback to regex method if API fails
                        return remove_answer_regex(teacher_thinking, teacher_answer)
        except Exception as e:
            logger.error(f"Error calling GPT-4o-mini: {e}")
            # Fallback to regex method if there's an exception
            return remove_answer_regex(teacher_thinking, teacher_answer)


def remove_answer_regex(thinking: str, answer: str) -> str:
    """
    Simple regex-based answer removal (fallback method).

    Args:
        thinking: The thinking trace
        answer: The answer to remove

    Returns:
        The thinking trace with answer removed
    """
    if not answer or not thinking:
        return thinking

    # Simple string replacement
    return re.sub(re.escape(answer), "", thinking)


def remove_answer_regex_no_commas(thinking: str, answer: str) -> str:
    """
    Answer removal with comma handling (used in trainers).

    Args:
        thinking: The thinking trace
        answer: The answer to remove

    Returns:
        The thinking trace with answer removed
    """
    if not answer or not thinking:
        return thinking

    # Remove commas from both thinking and answer
    thinking_no_commas = thinking.replace(',', '')
    answer_no_commas = answer.replace(',', '')

    # Check if the answer (without commas) is in the thinking (without commas)
    if answer_no_commas in thinking_no_commas:
        return thinking_no_commas.replace(answer_no_commas, '')

    return thinking


async def process_batch_remove_answers(items: List[Dict[str, Any]], api_key: str,
                                       max_concurrent: int = 50) -> List[Dict[str, Any]]:
    """
    Process a batch of items to remove answers using GPT-4o-mini concurrently.

    Args:
        items: List of items with 'teacher_thinking' and 'teacher_answer' fields
        api_key: OpenAI API key
        max_concurrent: Maximum number of concurrent API requests

    Returns:
        List of items with 'teacher_thinking_without_answer' field added
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_item(item):
        if item.get('teacher_response') is None:
            return None

        thinking_without_answer = await remove_answer_with_gpt4o_mini(
            item['teacher_thinking'],
            item['teacher_answer'],
            api_key,
            semaphore
        )
        item['teacher_thinking_without_answer'] = thinking_without_answer
        return item

    # Process all items concurrently with progress bar
    tasks = [process_item(item) for item in items]
    results = []
    for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks),
                           desc="Removing answers with GPT-4o-mini"):
        result = await coro
        if result is not None:
            results.append(result)

    return results


def remove_answers_batch(filtered_data: List[Dict[str, Any]], api_key: str = None,
                        use_llm: bool = True) -> List[Dict[str, Any]]:
    """
    Wrapper function to remove answers from a batch of items.

    Args:
        filtered_data: List of items to process
        api_key: OpenAI API key (will use OPENAI_API_KEY env var if not provided)
        use_llm: If True, use LLM-based removal; otherwise use regex

    Returns:
        List of items with answers removed
    """
    if not use_llm:
        # Use simple regex removal
        logger.info("Using regex method for answer removal")
        for item in filtered_data:
            if item.get("teacher_response") is not None:
                item['teacher_thinking_without_answer'] = remove_answer_regex(
                    item['teacher_thinking'],
                    item['teacher_answer']
                )
        return filtered_data

    # Use LLM-based removal
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found, falling back to regex method")
            # Fallback to simple regex
            for item in filtered_data:
                if item.get("teacher_response") is not None:
                    item['teacher_thinking_without_answer'] = remove_answer_regex(
                        item['teacher_thinking'],
                        item['teacher_answer']
                    )
            return filtered_data

    logger.info(f"Using GPT-4o-mini to remove answers from {len(filtered_data)} items")
    return asyncio.run(process_batch_remove_answers(filtered_data, api_key))


def remove_answer_sync(thinking: str, answer: str, use_llm: bool = False) -> str:
    """
    Synchronous wrapper to remove answer from a single thinking trace.

    Args:
        thinking: The thinking trace
        answer: The answer to remove
        use_llm: If True, use LLM-based removal; otherwise use regex

    Returns:
        The thinking trace with answer removed

    Note:
        For LLM-based removal, automatically uses OPENAI_API_KEY from environment.
    """
    if not use_llm:
        return remove_answer_regex(thinking, answer)

    # For single item LLM removal, wrap in batch processing
    # API key automatically picked up from environment in remove_answers_batch
    item = {
        'teacher_thinking': thinking,
        'teacher_answer': answer,
        'teacher_response': 'placeholder'
    }

    results = remove_answers_batch([item], api_key=None, use_llm=True)
    if results:
        return results[0].get('teacher_thinking_without_answer', thinking)

    return thinking
