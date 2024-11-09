# coding=utf-8

import base64
import os
from typing import Dict

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

from setting.models_provider.base_model_provider import MaxKBBaseModel


class QwenVLChatModel(MaxKBBaseModel, ChatOpenAI):

    @staticmethod
    def new_instance(model_type, model_name, model_credential: Dict[str, object], **model_kwargs):
        optional_params = MaxKBBaseModel.filter_optional_params(model_kwargs)
        chat_tong_yi = QwenVLChatModel(
            model=model_name,
            openai_api_key=model_credential.get('api_key'),
            openai_api_base='https://dashscope.aliyuncs.com/compatible-mode/v1',
            stream_options={"include_usage": True},
            model_kwargs=optional_params,
        )
        return chat_tong_yi

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



