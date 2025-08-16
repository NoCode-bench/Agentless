import time
from typing import Dict, Union

import anthropic
import openai
import tiktoken
import httpx
import os


def num_tokens_from_messages(message, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if isinstance(message, list):
        # use last message.
        num_tokens = len(encoding.encode(message[0]["content"]))
    else:
        num_tokens = len(encoding.encode(message))
    return num_tokens


def create_chatgpt_config(
    message: Union[str, list],
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "gpt-3.5-turbo",
) -> Dict:
    if "deepseek-reasoner" in model or "gemini" in model:
        max_tokens = 64000
    elif "qwen3" in model.lower() or "deepseek-chat" in model or "gpt-5-chat" in model.lower():
        max_tokens = 16000
    elif "gpt-5" in model.lower():
        max_tokens = 16000
    else:
        max_tokens = 128000
    if isinstance(message, list):
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [{"role": "system", "content": system_message}] + message,
        }
    else:
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": message},
            ],
        }
    return config


def handler(signum, frame):
    # swallow signum and frame
    raise Exception("end of time")

def request_chatgpt_with_batch(config, logger, base_url=None, max_retries=40, timeout=900):
    max_n_supported = 8
    n = config.get("n", 1)

    results = []
    if n > max_n_supported:
        batches = (n + max_n_supported - 1) // max_n_supported  # 向上取整
        for i in range(batches):
            batch_config = config.copy()
            batch_config["n"] = min(max_n_supported, n - i * max_n_supported)
            logger.info(f"Batch {i+1}/{batches}: 请求 {batch_config['n']} 个结果")
            ret = request_chatgpt_engine(batch_config, logger, base_url, max_retries, timeout)
            results.append(ret)
    else:
        ret = request_chatgpt_engine(config, logger, base_url, max_retries, timeout)
        results.append(ret)

    return results


def request_chatgpt_engine(config, logger, base_url=None, max_retries=40, timeout=900):
    ret = None
    retries = 0

    client = openai.OpenAI(base_url=os.getenv("OPENAI_API_BASE"), timeout=httpx.Timeout(900.0))

    while ret is None and retries < max_retries:
        try:
            # Attempt to get the completion
            logger.info("Creating API request")

            ret = client.chat.completions.create(**config)

        except openai.OpenAIError as e:
            if isinstance(e, openai.BadRequestError):
                msg = str(e)
                if "max_prompt_tokens" in msg or "max_total_tokens" in msg  or "maximum context length" in msg or"input length" in msg:
                    logger.info("400 BadRequest (tokens exceed limit). Stop retries, return empty string.")
                    ret = None
                    break

            if isinstance(e, openai.BadRequestError):
                logger.info("Request invalid")
                print(e)
                logger.info(e)
                raise Exception("Invalid API Request")
            elif isinstance(e, openai.RateLimitError):
                print("Rate limit exceeded. Waiting...")
                logger.info("Rate limit exceeded. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(10 * retries)
            elif isinstance(e, openai.APIConnectionError):
                print("API connection error. Waiting...")
                logger.info("API connection error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            elif isinstance(e, openai.ConflictError) or getattr(e, "status_code", None) == 409:
                logger.info("409 conflict (payload too long). Stop retries, return empty string.")
                ret = None         
                break
            else:
                print("Unknown error. Waiting...")
                logger.info("Unknown error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(1)

        retries += 1

    logger.info(f"API response {ret}")
    return ret

def create_gpt5_config(
    message: Union[str, list],
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "gpt-3.5-turbo",
) -> Dict:

    max_tokens = 128000
    if isinstance(message, list):
        config = {
            "model": model,
            "max_completion_tokens": max_tokens,
            "temperature": 1.0,
            "n": batch_size,
            "input": [{"role": "developer", "content": system_message}] + message,
            "reasoning_effort": "medium",
            "verbosity": "medium"
            
        }
    else:
        config = {
            "model": model,
            "max_completion_tokens": max_tokens,
            "temperature": 1.0,
            "n": batch_size,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": message},
            ],
            "reasoning_effort": "medium",
            "verbosity": "medium"
        }
    return config


def create_anthropic_config(
    message: str,
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "claude-2.1",
    tools: list = None,
) -> Dict:
    max_tokens = 32000
    if isinstance(message, list):
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": message,
        }
    else:
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": message}]},
            ],
        }

    if tools:
        config["tools"] = tools

    return config


def request_anthropic_engine(
    config, logger, max_retries=40, timeout=500, prompt_cache=False
):
    ret = None
    retries = 0

    client = anthropic.Anthropic()

    while ret is None and retries < max_retries:
        try:
            start_time = time.time()
            if prompt_cache:
                # following best practice to cache mainly the reused content at the beginning
                # this includes any tools, system messages (which is already handled since we try to cache the first message)
                config["messages"][0]["content"][0]["cache_control"] = {
                    "type": "ephemeral"
                }
                ret = client.beta.prompt_caching.messages.create(**config)
            else:
                ret = client.messages.create(**config)
        except Exception as e:
            logger.error("Unknown error. Waiting...", exc_info=True)
            # Check if the timeout has been exceeded
            if time.time() - start_time >= timeout:
                logger.warning("Request timed out. Retrying...")
            else:
                logger.warning("Retrying after an unknown error...")
            time.sleep(10 * retries)
        retries += 1

    return ret
