"""Prompt templates. Schema lives on Target; scalar.py and joint.py only supply wording."""

from argparse import Namespace
from typing import Any, Callable

from sampling.targets import get_target
from sampling.templates.joint import create_continuation_template as create_joint_continuation
from sampling.templates.joint import create_template as create_joint_template
from sampling.templates.scalar import create_continuation_template as create_scalar_continuation
from sampling.templates.scalar import create_template as create_scalar_template


def create_template_and_schema(
    args: Namespace, method: str, continuation: bool = False
) -> tuple[Callable[..., str], dict[str, Any]]:
    target = get_target(args.target)
    schema = target.object_schema(args, method)

    if target.is_joint:
        if continuation:
            template = create_joint_continuation(args)
        else:
            template = create_joint_template(method, args)
    else:
        if continuation:
            template = create_scalar_continuation(args)
        else:
            template = create_scalar_template(method, args)
    return template, schema
