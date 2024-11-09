import base64
import os
from typing import Dict

from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI

from common.config.tokenizer_manage_config import TokenizerManage
from setting.models_provider.base_model_provider import MaxKBBaseModel


def custom_get_token_ids(text: str):
    tokenizer = TokenizerManage.get_tokenizer()
    return tokenizer.encode(text)


class OpenAIImage(MaxKBBaseModel, ChatOpenAI):

    @staticmethod
    def new_instance(model_type, model_name, model_credential: Dict[str, object], **model_kwargs):
        optional_params = MaxKBBaseModel.filter_optional_params(model_kwargs)
        return OpenAIImage(
            model=model_name,
            openai_api_base=model_credential.get('api_base'),
            openai_api_key=model_credential.get('api_key'),
            stream_options={"include_usage": True},
            **optional_params,
        )

    @staticmethod
    def generate_message(prompt: str, image) -> list[HumanMessage]:
        if image is not None:
            base64_image = base64.b64encode(image.get_byte()).decode("utf-8")
            return [HumanMessage(
                content=[
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{base64_image}'}},
                ])]
        return [HumanMessage(prompt)]
