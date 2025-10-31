import json
from dataclasses import dataclass
from typing import Dict, Iterable, List

import bpy

try:
    import requests
except ImportError:  # pragma: no cover - Blender bundles requests, but guard just in case
    requests = None

from . import retargeting


class AIBoneMatchingError(Exception):
    """Base class for AI bone matching errors."""


class AIBoneMatchingConfigurationError(AIBoneMatchingError):
    """Raised when the AI integration is not configured correctly."""


class AIBoneMatchingRequestError(AIBoneMatchingError):
    """Raised when the AI request fails to reach the endpoint."""


class AIBoneMatchingResponseError(AIBoneMatchingError):
    """Raised when the AI response cannot be parsed."""


@dataclass(frozen=True)
class BoneItem:
    """Lightweight representation of a retargeting bone list entry."""

    bone_name_source: str
    bone_name_target: str
    bone_name_key: str


def _collect_bone_structure(armature: bpy.types.Object) -> List[Dict[str, object]]:
    bones: List[Dict[str, object]] = []
    if not armature or not armature.pose:
        return bones

    for pose_bone in armature.pose.bones:
        bones.append({
            'name': pose_bone.name,
            'parent': pose_bone.parent.name if pose_bone.parent else None,
            'children': [child.name for child in pose_bone.children],
            'head_local': [round(coord, 5) for coord in pose_bone.head],
            'tail_local': [round(coord, 5) for coord in pose_bone.tail],
        })

    return bones


def _build_messages(context_data: Dict[str, object]) -> List[Dict[str, str]]:
    json_context = json.dumps(context_data, ensure_ascii=False, indent=2)

    system_message = (
        "You are an expert technical rigger that assists with animation retargeting between Blender armatures. "
        "Your task is to match source bones to the most appropriate target bones. "
        "Always reply with valid JSON only, without explanations."
    )
    user_message = (
        "Match the bones listed in `retargeting_items` from the source armature to the best fitting bones in the target armature.\n"
        "Only use target bone names that appear in `target_armature.bones`.\n"
        "Consider the provided naming hints (`key`) and the hierarchy information.\n"
        "Return strict JSON in the following format:\n"
        "{\n"
        "  \"assignments\": [\n"
        "    {\"source\": \"<source bone>\", \"target\": \"<target bone>\"}\n"
        "  ]\n"
        "}\n"
        "Include every source bone that you can confidently map.\n"
        "Omit a source bone if there is no reasonable match.\n"
        "Context data:\n````json\n"
        f"{json_context}\n````"
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def _prepare_context(bone_items: Iterable[BoneItem]) -> Dict[str, object]:
    armature_source = retargeting.get_source_armature()
    armature_target = retargeting.get_target_armature()

    retargeting_items = []
    for item in bone_items:
        retargeting_items.append({
            'source': item.bone_name_source,
            'current_target': item.bone_name_target,
            'key': item.bone_name_key,
        })

    return {
        'source_armature': {
            'name': armature_source.name if armature_source else '',
            'bones': _collect_bone_structure(armature_source),
        },
        'target_armature': {
            'name': armature_target.name if armature_target else '',
            'bones': _collect_bone_structure(armature_target),
        },
        'retargeting_items': retargeting_items,
    }


def _post_request(endpoint: str, api_key: str, model: str, messages: List[Dict[str, str]]) -> Dict[str, object]:
    if requests is None:
        raise AIBoneMatchingConfigurationError('The "requests" module is required for AI bone matching but is not available.')

    headers = {
        'Content-Type': 'application/json',
    }
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    payload = {
        'model': model,
        'messages': messages,
        'temperature': 0.1,
        'max_tokens': 800,
    }

    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=60)
    except requests.exceptions.RequestException as exc:
        raise AIBoneMatchingRequestError(str(exc)) from exc

    if response.status_code >= 400:
        # Try to include a short hint from the response content for easier debugging
        hint = response.text
        if len(hint) > 180:
            hint = hint[:180] + '…'
        raise AIBoneMatchingRequestError(
            f'Endpoint returned HTTP {response.status_code}: {hint}'
        )

    try:
        return response.json()
    except ValueError as exc:
        raise AIBoneMatchingResponseError('Failed to decode JSON response from the AI endpoint.') from exc


def _extract_assignments(response_json: Dict[str, object]) -> Dict[str, str]:
    choices = response_json.get('choices')
    if not choices:
        raise AIBoneMatchingResponseError('The AI response does not contain any choices.')

    message = choices[0].get('message') if isinstance(choices[0], dict) else None
    if not message or 'content' not in message:
        raise AIBoneMatchingResponseError('The AI response is missing the message content.')

    content = message['content']
    if not isinstance(content, str):
        raise AIBoneMatchingResponseError('Unexpected message content format received from the AI endpoint.')

    content = content.strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise AIBoneMatchingResponseError('The AI response is not valid JSON. Enable deterministic mode or try again.') from exc

    assignments = parsed.get('assignments')
    if assignments is None:
        return {}

    if not isinstance(assignments, list):
        raise AIBoneMatchingResponseError('The AI response "assignments" field must be a list.')

    result: Dict[str, str] = {}
    for entry in assignments:
        if not isinstance(entry, dict):
            continue
        source = entry.get('source') or entry.get('source_bone')
        target = entry.get('target') or entry.get('target_bone')
        if isinstance(source, str) and isinstance(target, str):
            result[source] = target

    return result


def match_bones(scene: bpy.types.Scene, bone_items: Iterable[BoneItem]) -> Dict[str, str]:
    if not scene.rsl_retargeting_ai_enabled:
        raise AIBoneMatchingConfigurationError('Enable AI Bone Matching before requesting assignments.')

    endpoint = (scene.rsl_retargeting_ai_endpoint or '').strip()
    if not endpoint:
        raise AIBoneMatchingConfigurationError('Set the AI endpoint URL in the AI Bone Matching settings.')

    model = (scene.rsl_retargeting_ai_model or '').strip()
    if not model:
        raise AIBoneMatchingConfigurationError('Set the AI model name before requesting matches.')

    api_key = scene.rsl_retargeting_ai_api_key or ''

    armature_source = retargeting.get_source_armature()
    armature_target = retargeting.get_target_armature()

    if not armature_source or not armature_target:
        raise AIBoneMatchingConfigurationError('Select both source and target armatures before using AI Bone Matching.')

    context_data = _prepare_context(bone_items)
    messages = _build_messages(context_data)
    response_json = _post_request(endpoint, api_key, model, messages)
    return _extract_assignments(response_json)
