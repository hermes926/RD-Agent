import copyreg
import os
from typing import Any, Literal, Optional, Type, TypedDict, Union, cast

import numpy as np
from litellm import (
    BadRequestError,
    completion,
    completion_cost,
    embedding,
    get_model_info,
    supports_function_calling,
    supports_response_schema,
    token_counter,
)
from pydantic import BaseModel

from rdagent.log import LogColors
from rdagent.log import rdagent_logger as logger
from rdagent.oai.backend.base import APIBackend
from rdagent.oai.llm_conf import LLMSettings

# Attempt to import Tinker SDK components
try:
    import tinker
    import re
    from tinker import types as tinker_types
    from tinker_cookbook import renderers, tokenizer_utils
    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False


# NOTE: Patching! Otherwise, the exception will call the constructor and with following error:
# `BadRequestError.__init__() missing 2 required positional arguments: 'model' and 'llm_provider'`
def _reduce_no_init(exc: Exception) -> tuple:
    cls = exc.__class__
    return (cls.__new__, (cls,), exc.__dict__)


# suppose you want to apply this to MyError
copyreg.pickle(BadRequestError, _reduce_no_init)


class LiteLLMSettings(LLMSettings):

    class Config:
        env_prefix = "LITELLM_"
        """Use `LITELLM_` as prefix for environment variables"""

    # Tinker specific settings
    use_tinker: bool = False
    tinker_base_model: str = "Qwen/Qwen3-30B-A3B"  # Example default
    tinker_renderer_name: str = "qwen3" # Must match the model family (e.g. 'qwen3', 'llama3')
    tinker_api_key: Optional[str] = None


LITELLM_SETTINGS = LiteLLMSettings()
ACC_COST = 0.0


class LiteLLMAPIBackend(APIBackend):
    """
    Implementation of APIBackend interface that supports both LiteLLM and Tinker.
    """

    _has_logged_settings: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not self.__class__._has_logged_settings:
            logger.info(f"{LITELLM_SETTINGS}")
            logger.log_object(LITELLM_SETTINGS.model_dump(), tag="LITELLM_SETTINGS")
            self.__class__._has_logged_settings = True
        
        self.tinker_client_initialized = False
        if LITELLM_SETTINGS.use_tinker:
            if not TINKER_AVAILABLE:
                logger.error("Tinker is enabled in settings but 'tinker' or 'tinker_cookbook' python packages are not installed.")
            else:
                self._init_tinker_client()

        super().__init__(*args, **kwargs)

    def _init_tinker_client(self):
        """Initialize Tinker ServiceClient, SamplingClient, Tokenizer, and Renderer."""
        try:
            api_key = LITELLM_SETTINGS.tinker_api_key or os.environ.get("TINKER_API_KEY")
            if not api_key:
                logger.warning("Tinker API key not found. Please set LITELLM_TINKER_API_KEY or TINKER_API_KEY env var.")
            
            # Set the env var for Tinker SDK if explicitly provided in settings
            if api_key:
                os.environ["TINKER_API_KEY"] = api_key

            self.service_client = tinker.ServiceClient()
            self.sampling_client = self.service_client.create_sampling_client(
                base_model=LITELLM_SETTINGS.tinker_base_model
            )
            
            logger.info(f"{LogColors.GREEN}Initializing Tinker Tokenizer & Renderer...{LogColors.END}")
            self.tokenizer = tokenizer_utils.get_tokenizer(LITELLM_SETTINGS.tinker_base_model)
            self.renderer = renderers.get_renderer(LITELLM_SETTINGS.tinker_renderer_name, self.tokenizer)
            self.tinker_client_initialized = True
            logger.info(f"{LogColors.GREEN}Tinker Client Initialized{LogColors.END} for model {LITELLM_SETTINGS.tinker_base_model}")
        except Exception as e:
            logger.error(f"Failed to initialize Tinker client: {e}")
            self.tinker_client_initialized = False

    def _calculate_token_from_messages(self, messages: list[dict[str, Any]]) -> int:
        """
        Calculate the token count from messages.
        Uses Tinker's renderer/tokenizer if Tinker is enabled, otherwise LiteLLM.
        """
        if LITELLM_SETTINGS.use_tinker and self.tinker_client_initialized:
            try:
                # Convert messages to Tinker's ModelInput (prompt) and count tokens
                prompt = self.renderer.build_generation_prompt(messages)
                # prompt.to_ints() returns the list of token IDs
                num_tokens = len(prompt.to_ints())
                logger.info(f"{LogColors.CYAN}Tinker Token count:{LogColors.END} {num_tokens}", tag="debug_tinker_token")
                return num_tokens
            except Exception as e:
                logger.warning(f"Tinker token counting failed, falling back to default: {e}")

        num_tokens = token_counter(
            model=LITELLM_SETTINGS.chat_model,
            messages=messages,
        )
        logger.info(f"{LogColors.CYAN}Token count: {LogColors.END} {num_tokens}", tag="debug_litellm_token")
        return num_tokens

    def _create_embedding_inner_function(self, input_content_list: list[str]) -> list[list[float]]:
        """
        Call the embedding function.
        Note: Tinker currently focuses on Training/Sampling. If Tinker adds an embedding endpoint,
        implementation can be added here. Defaults to LiteLLM for now.
        """
        if LITELLM_SETTINGS.use_tinker:
             logger.warning("Tinker does not explicitly support Embeddings API yet. Falling back to LiteLLM/OpenAI for embeddings.")

        model_name = LITELLM_SETTINGS.embedding_model
        logger.info(f"{LogColors.GREEN}Using emb model{LogColors.END} {model_name}", tag="debug_litellm_emb")
        if LITELLM_SETTINGS.log_llm_chat_content:
            logger.info(
                f"{LogColors.MAGENTA}Creating embedding{LogColors.END} for: {input_content_list}",
                tag="debug_litellm_emb",
            )
        response = embedding(
            model=model_name,
            input=input_content_list,
        )
        response_list = [data["embedding"] for data in response.data]
        return response_list

    class CompleteKwargs(TypedDict):
        model: str
        temperature: float
        max_tokens: int | None
        reasoning_effort: Literal["low", "medium", "high"] | None

    def get_complete_kwargs(self) -> CompleteKwargs:
        """
        return several key settings for completion
        getting these values from settings makes it easier to adapt to backend calls in agent systems.
        """
        # Call LiteLLM completion
        model = LITELLM_SETTINGS.chat_model
        temperature = LITELLM_SETTINGS.chat_temperature
        max_tokens = LITELLM_SETTINGS.chat_max_tokens
        reasoning_effort = LITELLM_SETTINGS.reasoning_effort

        if LITELLM_SETTINGS.chat_model_map:
            for t, mc in LITELLM_SETTINGS.chat_model_map.items():
                if t in logger._tag:
                    model = mc["model"]
                    if "temperature" in mc:
                        temperature = float(mc["temperature"])
                    if "max_tokens" in mc:
                        max_tokens = int(mc["max_tokens"])
                    if "reasoning_effort" in mc:
                        if mc["reasoning_effort"] in ["low", "medium", "high"]:
                            reasoning_effort = cast(Literal["low", "medium", "high"], mc["reasoning_effort"])
                        else:
                            reasoning_effort = None
                    break
        return self.CompleteKwargs(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

    def _tinker_sample(
        self,
        messages: list[dict[str, Any]],
        **kwargs
    ) -> tuple[str, str | None]:
        """
        Internal method to handle Tinker sampling (inference).
        Includes logic to strip <think>...</think> tags from reasoning models.
        """
        try:
            # 1. Render prompt
            prompt = self.renderer.build_generation_prompt(messages)
            
            # 2. Prepare Sampling Params
            default_stops = self.renderer.get_stop_sequences()
            stop_sequences = kwargs.get("stop") or default_stops
            
            max_tokens = kwargs.get("max_tokens", LITELLM_SETTINGS.chat_max_tokens or 1024)
            temperature = kwargs.get("temperature", LITELLM_SETTINGS.chat_temperature)
            
            sampling_params = tinker_types.SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_sequences
            )

            # 3. Call Sample
            if LITELLM_SETTINGS.log_llm_chat_content:
                logger.info(f"{LogColors.GREEN}Using Tinker model{LogColors.END} {LITELLM_SETTINGS.tinker_base_model}", tag="tinker_sample")

            future = self.sampling_client.sample(
                prompt=prompt,
                sampling_params=sampling_params,
                num_samples=1
            )
            
            # 4. Wait for result
            result = future.result()
            
            # 5. Parse output
            output_tokens = result.sequences[0].tokens
            message_data, parse_success = self.renderer.parse_response(output_tokens)
            
            if isinstance(message_data, dict) or hasattr(message_data, "__getitem__"):
                content = message_data["content"]
            else:
                content = getattr(message_data, "content", str(message_data))

            # --- FIX: STRIP THINKING TAGS ---
            if "<think>" in content and "</think>" in content:
                # Log the thinking process for debugging purposes before stripping it
                thinking_content = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                if thinking_content and LITELLM_SETTINGS.log_llm_chat_content:
                    logger.info(f"{LogColors.YELLOW}Thinking Process:{LogColors.END}\n{thinking_content.group(1)}", tag="reasoning")

                # Remove the block
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            if "</think>" in content:
                # Log the discarded thinking process for debugging
                # We take everything *before* the tag as the thought
                raw_thought = content.split("</think>")[0]
                
                # Clean up potential opening tag for the log
                clean_log_thought = raw_thought.replace("<think>", "").strip()
                
                if LITELLM_SETTINGS.log_llm_chat_content:
                     logger.info(f"{LogColors.YELLOW}Thinking Process:{LogColors.END}\n{clean_log_thought}", tag="reasoning")

                # The actual content is everything *after* the closing tag
                content = content.split("</think>")[-1].strip()

            finish_reason = "stop" 
            
            if LITELLM_SETTINGS.log_llm_chat_content:
                logger.info(
                    f"{LogColors.BLUE}assistant (Tinker - Cleaned):{LogColors.END}\n{content}", tag="llm_messages"
                )
            
            return content, finish_reason

        except Exception as e:
            logger.error(f"Error during Tinker sampling: {e}")
            raise e

    def _create_chat_completion_inner_function(  # type: ignore[no-untyped-def] # noqa: C901, PLR0912, PLR0915
        self,
        messages: list[dict[str, Any]],
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        *args,
        **kwargs,
    ) -> tuple[str, str | None]:
        """
        Call the chat completion function.
        Dispatches to Tinker if `LITELLM_SETTINGS.use_tinker` is True.
        """
        
        # --- Tinker Branch ---
        if LITELLM_SETTINGS.use_tinker and self.tinker_client_initialized:
             complete_kwargs = self.get_complete_kwargs()
             # Merge kwargs
             tinker_kwargs = {**complete_kwargs, **kwargs}
             return self._tinker_sample(messages, **tinker_kwargs)
        # ---------------------

        if response_format and not supports_response_schema(model=LITELLM_SETTINGS.chat_model):
            # Deepseek will enter this branch
            logger.warning(
                f"{LogColors.YELLOW}Model {LITELLM_SETTINGS.chat_model} does not support response schema, ignoring response_format argument.{LogColors.END}",
                tag="llm_messages",
            )
            response_format = None

        if response_format:
            kwargs["response_format"] = response_format

        if LITELLM_SETTINGS.log_llm_chat_content:
            logger.info(self._build_log_messages(messages), tag="llm_messages")

        complete_kwargs = self.get_complete_kwargs()
        model = complete_kwargs["model"]

        response = completion(
            messages=messages,
            stream=LITELLM_SETTINGS.chat_stream,
            max_retries=0,
            **complete_kwargs,
            **kwargs,
        )
        if LITELLM_SETTINGS.log_llm_chat_content:
            logger.info(f"{LogColors.GREEN}Using chat model{LogColors.END} {model}", tag="llm_messages")

        if LITELLM_SETTINGS.chat_stream:
            if LITELLM_SETTINGS.log_llm_chat_content:
                logger.info(f"{LogColors.BLUE}assistant:{LogColors.END}", tag="llm_messages")
            content = ""
            finish_reason = None
            for message in response:
                if message["choices"][0]["finish_reason"]:
                    finish_reason = message["choices"][0]["finish_reason"]
                if "content" in message["choices"][0]["delta"]:
                    chunk = (
                        message["choices"][0]["delta"]["content"] or ""
                    )  # when finish_reason is "stop", content is None
                    content += chunk
                    if LITELLM_SETTINGS.log_llm_chat_content:
                        logger.info(LogColors.CYAN + chunk + LogColors.END, raw=True, tag="llm_messages")
            if LITELLM_SETTINGS.log_llm_chat_content:
                logger.info("\n", raw=True, tag="llm_messages")
        else:
            content = str(response.choices[0].message.content)
            finish_reason = response.choices[0].finish_reason
            finish_reason_str = (
                f"({LogColors.RED}Finish reason: {finish_reason}{LogColors.END})"
                if finish_reason and finish_reason != "stop"
                else ""
            )
            if LITELLM_SETTINGS.log_llm_chat_content:
                logger.info(
                    f"{LogColors.BLUE}assistant:{LogColors.END} {finish_reason_str}\n{content}", tag="llm_messages"
                )

        global ACC_COST
        try:
            cost = completion_cost(model=model, messages=messages, completion=content)
        except Exception as e:
            logger.warning(f"Cost calculation failed for model {model}: {e}. Skip cost statistics.")
            cost = np.nan
        else:
            ACC_COST += cost
            if LITELLM_SETTINGS.log_llm_chat_content:
                logger.info(
                    f"Current Cost: ${float(cost):.10f}; Accumulated Cost: ${float(ACC_COST):.10f}; {finish_reason=}",
                )

        prompt_tokens = token_counter(model=model, messages=messages)
        completion_tokens = token_counter(model=model, text=content)
        logger.log_object(
            {
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost,
                "accumulated_cost": ACC_COST,
            },
            tag="token_cost",
        )
        return content, finish_reason

    def supports_response_schema(self) -> bool:
        """
        Check if the backend supports function calling
        """
        # Tinker generally supports structured output via strict renderers/grammars, 
        # but for basic compatibility we check settings or base model.
        if LITELLM_SETTINGS.use_tinker:
             return False # Assume False for basic Tinker integration unless specific renderer supports it
             
        return supports_response_schema(model=LITELLM_SETTINGS.chat_model) and LITELLM_SETTINGS.enable_response_schema

    @property
    def chat_token_limit(self) -> int:
        """Suggest an input token limit, ensuring enough space in the context window for the maximum output tokens."""
        try:
            model_info = get_model_info(LITELLM_SETTINGS.chat_model)
            if model_info is None:
                return super().chat_token_limit

            max_input = model_info.get("max_input_tokens")
            max_output = model_info.get("max_output_tokens")

            if max_input is None or max_output is None:
                return super().chat_token_limit

            max_input_tokens = max_input - max_output
            return max_input_tokens
        except Exception as e:
            return super().chat_token_limit