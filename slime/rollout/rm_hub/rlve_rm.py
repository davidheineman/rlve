from Gym.environment import VerifiableEnvironment
from Gym.environments import identifier2environment
from typing import Dict, Any



def rlve_rm(args, environment : str, config : Dict, response : str) -> Dict[str, Any] :
    if args.answer_marker_type == r"\boxed{}" :
        answer_markers = (r"\boxed{", r"}")
        assert args.custom_prompt_preprocessor in ("ChatTemplate_NoSystemPrompt", "ChatTemplate_BoxedAnswer", )
    elif args.answer_marker_type == r"<answer></answer>" :
        answer_markers = (r"<answer>", r"</answer>")
        assert args.custom_prompt_preprocessor in ("TinyZero", )
    else :
        raise NotImplementedError(f"Answer marker type {args.answer_marker_type} not implemented.")

    problem : VerifiableEnvironment = identifier2environment[environment](answer_markers = answer_markers)
    problem.set_config(config)
    return problem.verifier(response)