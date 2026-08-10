"""Shared config.json models and shape-safe loading boundary."""

import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ConfigWarning:
    field: str
    message: str


@dataclass(frozen=True)
class AppConfigLoadResult:
    data: dict
    warnings: tuple[ConfigWarning, ...]
    needs_backup: bool = False


class LLMConfig(BaseModel):
    base_url: str
    api_key: str
    model_name: str


class TTSConfig(BaseModel):
    mode: Literal["local", "external"] = "local"
    url: str = "http://127.0.0.1:7860"
    device: str = "auto"
    language: str = "English"
    parallel_workers: int = Field(default=2, ge=1)
    batch_seed: Optional[int] = None
    compile_codec: bool = False
    max_new_tokens: int = Field(default=2048, ge=256, le=8192)
    sub_batch_enabled: bool = True
    sub_batch_min_size: int = Field(default=4, ge=1)
    sub_batch_ratio: float = Field(default=5.0, ge=1)
    sub_batch_max_items: int = Field(default=0, ge=0)
    batch_group_by_type: bool = False
    pause_between_speakers_ms: int = Field(default=500, ge=0)
    pause_same_speaker_ms: int = Field(default=250, ge=0)


class ThreePassModelProfile(BaseModel):
    chunk_size: Optional[int] = Field(default=None, ge=500, le=12000)
    segment_temperature: Optional[float] = Field(default=None, ge=0, le=2)
    attribute_temperature: Optional[float] = Field(default=None, ge=0, le=2)
    instruct_temperature: Optional[float] = Field(default=None, ge=0, le=2)
    segment_output_ratio: Optional[float] = Field(default=None, ge=1.25, le=6.0)
    presegment_quotes: Optional[bool] = None


class GenerationConfig(BaseModel):
    chunk_size: int = Field(default=3000, ge=500)
    max_tokens: int = Field(default=4096, ge=256)
    temperature: float = Field(default=0.6, ge=0, le=2)
    top_p: float = Field(default=0.8, ge=0, le=1)
    top_k: int = Field(default=0, ge=0, le=200)
    min_p: float = Field(default=0, ge=0, le=1)
    presence_penalty: float = Field(default=0.0, ge=-2, le=2)
    banned_tokens: List[str] = Field(default_factory=list)
    merge_narrators: bool = False
    context_rescue_windows: List[Annotated[int, Field(gt=0)]] = Field(
        default_factory=lambda: [2000, 4000, 6000], min_length=1)
    context_rescue_retries: int = Field(default=2, ge=0)
    three_pass_segment_temperature: float = Field(default=0.1, ge=0, le=2)
    three_pass_attribute_temperature: float = Field(default=0.1, ge=0, le=2)
    three_pass_instruct_temperature: float = Field(default=0.1, ge=0, le=2)
    three_pass_segment_output_ratio: float = Field(default=3.0, ge=1.25, le=6.0)
    three_pass_chunk_size: int = Field(default=3000, ge=500, le=12000)
    three_pass_presegment_quotes: bool = True
    three_pass_model_profiles: Dict[str, ThreePassModelProfile] = Field(default_factory=dict)


class PromptConfig(BaseModel):
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    review_system_prompt: Optional[str] = None
    review_user_prompt: Optional[str] = None
    persona_system_prompt: Optional[str] = None
    persona_user_prompt: Optional[str] = None
    persona_advanced_prompt: Optional[str] = None


class AppConfig(BaseModel):
    llm: LLMConfig
    llm_mode: Literal["local", "remote"] = "local"
    llm_local: Optional[LLMConfig] = None
    llm_remote: Optional[LLMConfig] = None
    llm_remote_ssh: Optional[str] = None
    tts: TTSConfig
    prompts: Optional[PromptConfig] = None
    generation: Optional[GenerationConfig] = None


_DICT_SECTIONS = {"llm": LLMConfig, "tts": TTSConfig}
_OPTIONAL_DICT_SECTIONS = {
    "prompts": PromptConfig,
    "generation": GenerationConfig,
    "llm_local": LLMConfig,
    "llm_remote": LLMConfig,
}


def _get_field_adapter(model: type[BaseModel], field_name: str) -> TypeAdapter:
    field = model.model_fields[field_name]
    annotation = field.annotation
    if field.metadata:
        annotation = Annotated.__class_getitem__((annotation, *field.metadata))
    return TypeAdapter(annotation)


def _validate_present_fields(section_name: str, data: dict,
                             model: type[BaseModel], warnings: list) -> dict:
    validated = dict(data)
    for field_name in model.model_fields:
        if field_name not in validated:
            continue
        try:
            validated[field_name] = _get_field_adapter(model, field_name).validate_python(
                validated[field_name]
            )
        except ValidationError:
            path = f"{section_name}.{field_name}"
            warning = ConfigWarning(path, "Invalid stored value ignored")
            warnings.append(warning)
            logger.warning("Invalid stored config value '%s', ignoring it", path)
            del validated[field_name]
    return validated


def load_app_config_result(path: str) -> AppConfigLoadResult:
    """Load and sanitize present known values without defaults or file writes."""
    warnings = []
    needs_backup = False
    if not os.path.exists(path):
        return AppConfigLoadResult({}, ())
    try:
        with open(path, "r", encoding="utf-8") as config_file:
            loaded = json.load(config_file)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Corrupted/unreadable JSON at %s, using defaults: %s", path, exc)
        return AppConfigLoadResult(
            {}, (ConfigWarning("$", "Configuration file could not be read"),), True
        )
    if not isinstance(loaded, dict):
        logger.warning("Invalid top-level config shape '%s', using defaults", type(loaded).__name__)
        return AppConfigLoadResult(
            {}, (ConfigWarning("$", "Configuration must be a JSON object"),), True
        )
    config = dict(loaded)

    for section, model in _DICT_SECTIONS.items():
        if section not in config:
            continue
        if not isinstance(config[section], dict):
            warnings.append(ConfigWarning(section, "Invalid section ignored"))
            needs_backup = True
            logger.warning("Invalid '%s' section in config, ignoring it", section)
            del config[section]
            continue
        config[section] = _validate_present_fields(section, config[section], model, warnings)

    for section, model in _OPTIONAL_DICT_SECTIONS.items():
        if config.get(section) is None:
            continue
        if not isinstance(config[section], dict):
            warnings.append(ConfigWarning(section, "Invalid section ignored"))
            needs_backup = True
            logger.warning("Invalid '%s' section in config, ignoring it", section)
            config[section] = None
            continue
        config[section] = _validate_present_fields(section, config[section], model, warnings)

    for field_name in ("llm_mode", "llm_remote_ssh"):
        if field_name not in config:
            continue
        try:
            config[field_name] = _get_field_adapter(AppConfig, field_name).validate_python(
                config[field_name]
            )
        except ValidationError:
            warnings.append(ConfigWarning(field_name, "Invalid stored value ignored"))
            logger.warning("Invalid stored config value '%s', ignoring it", field_name)
            del config[field_name]

    return AppConfigLoadResult(config, tuple(warnings), needs_backup)


def _to_plain(value):
    """Recursively convert validated models back to JSON-native data.

    WHY THIS EXISTS. The sanitizer validates some sections into pydantic
    models - `generation.three_pass_model_profiles` is a
    Dict[str, ThreePassModelProfile]. Ten modules load the config and write it
    back (the concurrency cache in llm_bench, personas, nicknames, review,
    project, ...), and every one of them raised
    "Object of type ThreePassModelProfile is not JSON serializable" as soon as
    a profile was configured. Running --dedupe-speakers on index18 hit it.

    three_pass_generate already worked around this locally with
    `as_profile_mapping`, whose docstring records the same class of crash. A
    per-module workaround leaves the other nine broken, so the conversion
    belongs here, where the data is produced.

    None-valued fields are dropped, preserving the semantics
    `as_profile_mapping` established: an unset profile key must fall through to
    the caller's default rather than override it with None.
    """
    if hasattr(value, "model_dump"):
        return {key: _to_plain(item)
                for key, item in value.model_dump().items()
                if item is not None}
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    return value


def load_app_config(path: str) -> dict:
    """Return the sanitized config data from :func:`load_app_config_result`.

    The result is JSON-native, so a caller may write it back unchanged.
    """
    return _to_plain(load_app_config_result(path).data)


def backup_damaged_app_config(path: str) -> str:
    """Create an atomic, metadata-preserving backup beside a damaged config."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup_path = f"{path}.damaged-{stamp}-{uuid.uuid4().hex[:8]}.bak"
    temp_path = f"{backup_path}.tmp"
    try:
        shutil.copy2(path, temp_path)
        os.replace(temp_path, backup_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    return backup_path
