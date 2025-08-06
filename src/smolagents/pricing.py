MODEL_PRICES = {
    # Prices per 1 million tokens
    # Chat models
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-2025-04-14": {"input": 2.00, "output": 8.00},
    "zhdq:gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-mini-2025-04-14": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4.1-nano-2025-04-14": {"input": 0.10, "output": 0.40},
    "gpt-4.5-preview": {"input": 75.00, "output": 150.00},
    "gpt-4.5-preview-2025-02-27": {"input": 75.00, "output": 150.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00},
    "gpt-4o-audio-preview": {"input": 2.50, "output": 10.00},
    "gpt-4o-audio-preview-2024-12-17": {"input": 2.50, "output": 10.00},
    "gpt-4o-realtime-preview": {"input": 5.00, "output": 20.00},
    "gpt-4o-realtime-preview-2024-12-17": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini-audio-preview": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini-audio-preview-2024-12-17": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini-realtime-preview": {"input": 0.60, "output": 2.40},
    "gpt-4o-mini-realtime-preview-2024-12-17": {"input": 0.60, "output": 2.40},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-2024-12-17": {"input": 15.00, "output": 60.00},
    "o1-pro": {"input": 150.00, "output": 600.00},
    "o1-pro-2025-03-19": {"input": 150.00, "output": 600.00},
    "o3": {"input": 10.00, "output": 40.00},
    "o3-2025-04-16": {"input": 10.00, "output": 40.00},
    "o4-mini": {"input": 1.10, "output": 4.40},
    "o4-mini-2025-04-16": {"input": 1.10, "output": 4.40},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o3-mini-2025-01-31": {"input": 1.10, "output": 4.40},
    "o1-mini": {"input": 1.10, "output": 4.40},
    "o1-mini-2024-09-12": {"input": 1.10, "output": 4.40},

    # Claude models
    "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},

    # Gemini models
    "gemini-2.5-pro-exp-03-25": {"input": 1.25, "output": 2.50},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40}, # Assuming "0.4美元/tokens" was a typo for M tokens like others
    "gemini-1.5-pro": {"input": 1.25, "output": 2.50},

    # Qwen models (prices converted from CNY to USD, 1 CNY = 0.14 USD)
    # qwq-plus (assuming qwen-plus): 0.0016 CNY/k input, 0.004 CNY/k output
    # Input: 1.6 CNY/M * 0.14 USD/CNY = 0.224 USD/M
    # Output: 4.0 CNY/M * 0.14 USD/CNY = 0.56 USD/M
    "qwen-plus": {"input": 0.224, "output": 0.56},

    # qwen3-235b-a22b
    # Thinking: 0.004 CNY/k input (0.56 USD/M), 0.012 CNY/k output (1.68 USD/M)
    "qwen3-235b-a22b-thinking": {"input": 0.56, "output": 1.68},
    # Non-thinking: 0.004 CNY/k input (0.56 USD/M), 0.04 CNY/k output (5.60 USD/M)
    "qwen3-235b-a22b-non-thinking": {"input": 0.56, "output": 5.60},

    # qwen3-32b
    # Thinking: 0.002 CNY/k input (0.28 USD/M), 0.008 CNY/k output (1.12 USD/M)
    "qwen3-32b-thinking": {"input": 0.28, "output": 1.12},
    # Non-thinking: 0.002 CNY/k input (0.28 USD/M), 0.02 CNY/k output (2.80 USD/M)
    "qwen3-32b-non-thinking": {"input": 0.28, "output": 2.80},

    # qwen3-30b-a3b
    # Thinking: 0.0015 CNY/k input (0.21 USD/M), 0.008 CNY/k output (1.12 USD/M)
    "qwen3-30b-a3b-thinking": {"input": 0.21, "output": 1.12},
    # Non-thinking: 0.006 CNY/k input (0.84 USD/M), 0.015 CNY/k output (2.10 USD/M)
    "qwen3-30b-a3b-non-thinking": {"input": 0.84, "output": 2.10},

    # Embedding models - price per 1 million tokens for total tokens
    "text-embedding-3-small": {"total": 0.02},
    "text-embedding-3-large": {"total": 0.13},
    "text-embedding-ada-002": {"total": 0.10},
}

def get_model_price(model_id: str):
    """
    Retrieves the price information for a given model ID.
    Handles common prefixes like 'zhdq:' by stripping them before lookup.
    """
    # Normalize model_id by taking the part after the last colon, if any.
    # This handles cases like "zhdq:gpt-4.1" -> "gpt-4.1"
    normalized_model_id = model_id.split(':')[-1]
    return MODEL_PRICES.get(normalized_model_id)

def calculate_cost(model_id: str, prompt_tokens: int, completion_tokens: int = 0, is_embedding: bool = False):
    """
    Calculates the cost for a given model, tokens, and type (chat/embedding).
    """
    price_info = get_model_price(model_id)
    if not price_info:
        return 0.0, 0.0, 0.0 # input_cost, output_cost, total_cost

    input_cost = 0.0
    output_cost = 0.0

    if is_embedding:
        # For embeddings, cost is based on total (prompt) tokens
        price_total_per_million = price_info.get("total")
        if price_total_per_million is not None:
            total_cost = (prompt_tokens / 1_000_000) * price_total_per_million
            return 0.0, 0.0, total_cost # Embeddings usually don't differentiate input/output cost this way
        return 0.0, 0.0, 0.0
    else:
        # For chat models
        price_input_per_million = price_info.get("input")
        price_output_per_million = price_info.get("output")

        if price_input_per_million is not None:
            input_cost = (prompt_tokens / 1_000_000) * price_input_per_million
        if price_output_per_million is not None and completion_tokens > 0:
            output_cost = (completion_tokens / 1_000_000) * price_output_per_million
        
        return input_cost, output_cost, input_cost + output_cost 