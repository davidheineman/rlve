import argparse
import os

from transformers import AutoConfig

from slime.backends.sglang_utils.arguments import add_sglang_arguments
from slime.backends.sglang_utils.arguments import validate_args as sglang_validate_args


def reset_arg(parser, name, **kwargs):
    """
    Reset the default value of a Megatron argument.
    :param parser: The argument parser.
    :param name: The name of the argument to reset.
    :param default: The new default value.
    """
    for action in parser._actions:
        if name in action.option_strings:
            if "default" in kwargs:
                action.default = kwargs["default"]
            break
    else:
        parser.add_argument(name, **kwargs)


def get_slime_extra_args_provider(add_custom_arguments=None):
    def add_slime_arguments(parser):
        # Ray
        def add_cluster_arguments(parser):
            parser.add_argument("--actor-num-nodes", type=int, default=1, help="Number of nodes for training actor")
            parser.add_argument(
                "--actor-num-gpus-per-node", type=int, default=8, help="Number of gpus per node for training actor"
            )

            parser.add_argument(
                "--rollout-num-gpus",
                type=int,
                default=None,
                help=(
                    "Number of GPUs for inference. Note that when using --colocate, "
                    "i.e. the training and the inference engines are on the same gpus, this param will be ignored and will be set as "
                    "actor_num_gpus_per_node * actor_num_nodes."
                ),
            )
            parser.add_argument(
                "--rollout-num-gpus-per-engine",
                type=int,
                default=1,
                help="Number of GPUs per inference engine, just like the tp_size in sglang.",
            )
            parser.add_argument(
                "--num-gpus-per-node",
                type=int,
                default=8,
                help=(
                    "Number of gpus per node for rollout."
                    "Notice: If you are going to use less than 8 gpus per node under colocate mode, you should set this number."
                ),
            )
            parser.add_argument(
                "--colocate",
                action="store_true",
                default=False,
                help=(
                    "Whether to colocate the inference engines and the actor. "
                    "Turning this on will also set --offload to true."
                ),
            )
            parser.add_argument(
                "--offload",
                action="store_true",
                default=False,
                help=(
                    "Whether to offload the rollout generator and training actor to CPU during training. "
                    "This will always be true when --colocate is set."
                ),
            )
            parser.add_argument("--opd-teacher-dir", type=str, default=None)
            parser.add_argument("--opd-teacher-gpus", type=int, default=0)
            parser.add_argument("--opd-teacher-cache-size", type=int, default=0,
                                help="Maximum teachers resident per scorer GPU; zero keeps the full bank on GPU.")
            parser.add_argument("--opd-student-eos-token-id", type=int, default=None,
                                help="Canonical student stop token for optional teacher EOS-label transport.")
            parser.add_argument("--opd-teacher-eos-token-id", type=int, default=None,
                                help="Teacher stop token to swap with the student's stop token when scoring OPD.")
            parser.add_argument("--opd-replay-fraction", type=float, default=0.0,
                                help="Fraction of student batch trajectories replaced by cached OPD traces.")
            parser.add_argument("--opd-replay-capacity-per-env", type=int, default=400)
            parser.add_argument("--opd-replay-clip-epsilon", type=float, default=0.2)
            parser.add_argument("--opd-replay-load", type=str, default=None,
                                help="Committed replay snapshot paired with the parent student checkpoint.")
            parser.add_argument("--early-stop-val-accuracy", type=float, default=None)
            parser.add_argument(
                "--early-stop-eval-names", nargs="+", default=["val_envs"],
                help="Eval dataset names whose combined exact-match accuracy controls early stopping.",
            )
            parser.add_argument("--eval-on-last-rollout", action="store_true")
            parser.add_argument("--early-stop-patience-evals", type=int, default=None)
            parser.add_argument("--early-stop-min-rollouts", type=int, default=0)
            parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
            parser.add_argument("--require-early-stop", action="store_true")

            reset_arg(parser, "--distributed-backend", type=str, default="nccl")
            reset_arg(parser, "--distributed-timeout-minutes", type=int, default=10)

            return parser

        # rollout
        def add_rollout_arguments(parser):
            parser.add_argument(
                "--hf-checkpoint",
                type=str,
                default=None,
                help=(
                    "The huggingface checkpoint of the trained model. "
                    "This is used to initialize sglang and also provide the tokenizer. "
                    "Note that, we will always update the parameters in sglang with that of megatron before training, "
                    "so you only need to provide a huggingface checkpoint that has the same architecture as the model you want to train. "
                    "It doesn't necessary need to contain the most up-to-date parameters."
                ),
            )
            parser.add_argument(
                "--model-name",
                type=str,
                default=None,
                help=(
                    "The name of the model, this is used to convert the megatron weights into huggingface format. "
                    "If not set, we will use `type(AutoConfig.from_pretrained(args.hf_checkpoint)).__name__.lower()` as model_name. "
                    "Also, sometimes this will help alleviate the bug that transformers cannot find certain model."
                ),
            )
            parser.add_argument(
                "--rollout-function-path",
                type=str,
                default="slime.rollout.sglang_rollout.generate_rollout",
                help=(
                    "Path to the rollout generation function."
                    "You should use this model to create your own custom rollout function, "
                    "and then set this to the path of your custom rollout function. "
                    "The signature of the function should be "
                    "`def generate_rollout(args, rollout_id, *, evaluation=False) -> list[list[Sample]]`"
                    "and within the output sample, you should at least set `tokens`, `response_length`, `reward` "
                    "and `truncated`."
                ),
            )
            parser.add_argument(
                "--rollout-temperature",
                type=float,
                default=1.0,
                help="the temperature for the inference engine during rollout.",
            )
            parser.add_argument(
                "--rollout-top-p", type=float, default=1.0, help="the top-p for the inference engine during rollout."
            )
            parser.add_argument(
                "--rollout-top-k", type=int, default=-1, help="the top-k for the inference engine during rollout."
            )
            parser.add_argument(
                "--rollout-max-context-len",
                type=int,
                default=None,
                help=(
                    "The maximum context size for the inference engine during rollout."
                    "It should no exceed the `max_position_embeddinds` in Huggingface model's `config.json`"
                ),
            )
            parser.add_argument(
                "--rollout-max-prompt-len",
                type=int,
                default=None,
                help=(
                    "The maximum length of the prompt for the inference engine during rollout. "
                    "If set, we will filter out the long prompts during initialization of the global dataset. "
                    "This is not recommended if the dataset is large."
                ),
            )
            parser.add_argument(
                "--rollout-max-response-len",
                type=int,
                default=1024,
                help=(
                    "The maximum length of the response for the inference engine during rollout. "
                    "It is basically `max_tokens` in sglang."
                ),
            )
            parser.add_argument(
                "--answer-forcing",
                action="store_true",
                help="After a response hits its generation cap, append a final-answer prefix and sample a short continuation (also during eval).",
            )
            parser.add_argument(
                "--answer-forcing-extra-tokens",
                type=int,
                default=128,
                help="Maximum sampled tokens after the forced final-answer prefix.",
            )
            parser.add_argument(
                "--rollout-skip-special-tokens",
                action="store_true",
                default=False,
                help=(
                    "Whether to skip special tokens in the response during rollout. "
                    "This is useful when you want to use the response as a prompt for the next rollout."
                ),
            )
            parser.add_argument(
                "--rollout-stop",
                type=str,
                nargs="+",
                default=None,
                help=(
                    "The stop words for the inference engine during rollout. "
                    "It can be a list of strings or a single string. "
                    "It may be hard to pass special tokens in command line, in that case rollout_stop_token_ids can be used."
                ),
            )
            parser.add_argument(
                "--rollout-stop-token-ids",
                type=int,
                nargs="+",
                default=None,
                help=(
                    "The stop token ids for the inference engine during rollout. "
                    "It can be a list of integers or a single integer."
                ),
            )
            parser.add_argument(
                "--rollout-shuffle",
                action="store_true",
                default=False,
                help=("Whether to shuffle the prompts during rollout."),
            )
            parser.add_argument(
                "--rollout-seed",
                type=int,
                default=42,
                help=(
                    "The seed for the random number generator during rollout. "
                    "This is used to shuffle the prompts and also for the random sampling of the prompts."
                ),
            )

            # sampling
            parser.add_argument(
                "--over-sampling-batch-size",
                type=int,
                default=None,
                help=(
                    "This defines the granularity of the sampling batch in the rollout function. "
                    "When the number of available samples falls below the target, a sampling "
                    "operation of size over_sampling_batch_size will be triggered."
                    "Regardless of whether partial rollout is used or filters are applied, "
                    "the sampling granularity is always determined by this value. "
                    "If this value is None, rollout_batch_size will be used as the default over_sampling_batch_size."
                ),
            )
            parser.add_argument(
                "--dynamic-sampling-filter-path",
                type=str,
                default=None,
                help=(
                    "This is the filter function for dynamic sampling. "
                    "It should be able to judge whether the result of a prompt should be selected or not."
                    "We will do dynamic filter for sampling as in DAPO. e.g. not all correct or all wrong samples."
                    "You could use `slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std` as an example."
                ),
            )
            parser.add_argument(
                "--dynamic-sampling-max-rejections",
                type=int,
                default=None,
                help="Accept unfiltered groups after this many rejected groups per rollout, avoiding an endless DAPO search.",
            )

            # partial rollout
            parser.add_argument(
                "--partial-rollout",
                action="store_true",
                default=False,
                help=(
                    "Whether to use partial rollout. "
                    "If set, the unfinished samples during dynamic sampling will be recycled back to data buffer. "
                    "This is useful for long responses."
                ),
            )
            parser.add_argument(
                "--custom-generate-function-path",
                type=str,
                default=None,
                help=(
                    "Only substitue the `def generate(args, sample, sampling_params)` function within the example rollout function. "
                    "This should be useful if you need to implement some special rollout logic, e.g. multi-turn, function calling."
                ),
            )

            parser.add_argument(
                "--buffer-filter-path",
                type=str,
                default=None,
                help=(
                    "Path to the buffer filter function. "
                    "It should be able to select the samples in the buffer. "
                    "The function should take list[list[Sample]] and return list[list[Sample]]."
                ),
            )
            # update weight
            parser.add_argument(
                "--update-weight-buffer-size",
                type=int,
                default=512 * 1024**2,
                help=(
                    "buffer size for update weight, in bytes. "
                    "This is used for updating weights by chunk and should be useful for MoE models."
                ),
            )
            parser.add_argument(
                "--update-weights-interval",
                type=int,
                default=1,
                help="Interval for updating the weights",
            )
            parser.add_argument(
                "--fixed-rollout-policy",
                action="store_true",
                help="Generate training rollouts with the unchanged initial policy while evaluating the student.",
            )
            parser.add_argument(
                "--keep-old-actor",
                action="store_true",
                help="Whether to keep the rollout model on training process",
            )

            parser.add_argument(
                "--rollout-data-postprocess-path",
                type=str,
                default=None,
                help=(
                    "The called after we have all the rollout data including log_probs. "
                    "It may be helpful for updating loss mask."
                ),
            )
            return parser

        # data
        def add_data_arguments(parser):
            # dataset
            # TODO: maybe add an num_epoch and calculate the num_rollout from buffer
            parser.add_argument(
                "--num-rollout",
                type=int,
                default=None,
                help="Number of rollout steps. Currently, we don't support passing num_epoch and calculate num_rollout from data size.",
            )
            parser.add_argument(
                "--num-epoch",
                type=int,
                default=None,
                help=(
                    "Number of epochs for the training. "
                    "This is used to calculate the number of rollout steps from the dataset size. "
                    "If set, we will calculate the number of rollout steps as `num_rollout = num_epoch * dataset_size // rollout_batch_size`."
                ),
            )

            parser.add_argument(
                "--disable-rollout-global-dataset",
                action="store_false",
                dest="rollout_global_dataset",
                help=(
                    "Whether to use a global dataset for rollout. "
                    "If set, the rollout will use the `--prompt-data` as the prompt dataset, "
                    "and the prompts for rollout will be sampled from the dataset. "
                    "If not set, you need to manage the data by your self."
                ),
            )

            parser.add_argument(
                "--prompt-data",
                type=str,
                default=None,
                help=(
                    "The path to the prompt data. "
                    "Currently we only support jsonl format, and each line should contains --input-key and --label-key, "
                    "which will be used as the prompt and the label respectively. "
                    "If you want to use a custom template, you can set --apply-chat-template to true, in that case, "
                    "the input should be the same structure as an openai message, e.g. [\{'role': 'user', 'content': 'blabla'\}]. "
                ),
            )
            parser.add_argument(
                "--program-tree-sampler-spec", type=str, default=None,
                help="JSON specification for fresh program-tree prompts sampled at each rollout step.",
            )
            parser.add_argument("--apply-chat-template", action="store_true", default=False)
            parser.add_argument("--input-key", type=str, default="input", help="JSON dataset key")
            parser.add_argument("--label-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument("--metadata-key", type=str, default="metadata", help="JSON dataset key")
            parser.add_argument(
                "--tool-key",
                type=str,
                default=None,
                help=(
                    "When need to add tools during apply_chat_template, you should provide the key for the tools in the prompt dataset."
                ),
            )

            parser.add_argument(
                "--start-rollout-id",
                type=int,
                default=None,
                help=(
                    "The starting rollout step, if not set, will try to load the step from --load when doing continue training, "
                    "otherwise will be set to 0, meaning training from start."
                ),
            )

            # batch sizes
            parser.add_argument(
                "--rollout-batch-size",
                type=int,
                required=True,
                help=(
                    "The number of prompts in each rollout step. "
                    "The total data returned should be rollout_batch_size * n_samples_per_prompt. "
                ),
            )
            parser.add_argument(
                "--n-samples-per-prompt", type=int, default=1, help="Number of responses for each prompt in generation"
            )

            # gbs of the training, note that the gbs is of sample, not of prompts,
            # so if you hope to train 1 step for each rollout, the global_bach_size should be set as
            # `rollout_batch_size * n_samples_per_prompt`.
            reset_arg(parser, "--global-batch-size", type=int, default=None)
            parser.add_argument(
                "--num-steps-per-rollout",
                type=int,
                default=None,
                help=(
                    "Number of steps per rollout, e.g. It is equivalent to setting gbs as "
                    "`rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout`."
                ),
            )
            parser.add_argument(
                "--train-reuse-epochs",
                type=int,
                default=1,
                help="Replay each scored rollout batch for this many full optimizer passes.",
            )
            parser.add_argument(
                "--rollout-anneal-thresholds", type=float, nargs="+", default=None,
                help="Validation accuracies at which to decrease rollouts per prompt.",
            )
            parser.add_argument(
                "--rollout-anneal-sizes", type=int, nargs="+", default=None,
                help="Rollouts per prompt to use at each validation accuracy threshold.",
            )
            # mbs for the training, will be ignored if `use_dynamic_batch_size` is set.
            reset_arg(parser, "--micro-batch-size", type=int, default=1)
            parser.add_argument(
                "--balance-data",
                action="store_true",
                default=False,
                help=(
                    "Balance the number of tokens between data parallel ranks with `karmarkar_karp` for verl. "
                    "Note that this may allocate the different response of the same prompt into different training steps."
                ),
            )

            parser.add_argument(
                "--use-dynamic-batch-size",
                action="store_true",
                default=False,
                help=(
                    "Because the sample length varies, to maximize the GPU utilization, "
                    "we will use the dynamic batch size to adjust the micro batch size according to the maximum number of tokens each gpu can run. "
                    "For example, if we have 3 samples, with the length of 100, 200, and 300, and the max_tokens_per_gpu is 300, when enabling "
                    "dynamic batch size, slime will make 2 micro batches, i.e. [100, 200], [300]."
                ),
            )
            parser.add_argument(
                "--max-tokens-per-gpu",
                type=int,
                default=None,
                help=(
                    "The maximum number of tokens per GPU for dynamic batch size. "
                    "Note that when enabling context parallel (CP), the max tokens per gpu should be around "
                    "`max_response_len // cp_size` instead of `max_response_len`."
                ),
            )
            parser.add_argument(
                "--log-probs-max-tokens-per-gpu",
                type=int,
                default=None,
                help=(
                    "The maximum number of tokens per GPU for calculating log probs. "
                    "This is used to calculate the log probs of the responses during rollout, "
                    "and should be set to a larger value than `max_tokens_per_gpu` if you want better performance. "
                ),
            )
            return parser

        def add_eval_arguments(parser):
            parser.add_argument(
                "--eval-function-path",
                type=str,
                default=None,
                help=(
                    "Path to the eval generation function."
                    "If not set, we will use rollout_function_path as the default. "
                ),
            )

            # change the default value of eval_interval from Megatron to None
            reset_arg(parser, "--eval-interval", type=int, default=None)

            parser.add_argument(
                "--eval-prompt-data",
                type=str,
                default=None,
                nargs="+",
                help=(
                    "Path to the evaluation prompt data, "
                    "should first input the name of the eval dataset and then the path, e.g. "
                    "aime /path/to/aime.jsonl"
                ),
            )

            # The following keys are used to override the rollout version during eval.
            parser.add_argument("--eval-input-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument("--eval-label-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument("--eval-tool-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument(
                "--n-samples-per-eval-prompt",
                type=int,
                default=1,
                help="number of responses for each prompt in generation",
            )
            parser.add_argument("--eval-temperature", type=float, default=None)
            parser.add_argument("--eval-top-p", type=float, default=None)
            parser.add_argument("--eval-top-k", type=int, default=None)
            parser.add_argument("--eval-max-response-len", type=int, default=None)
            parser.add_argument("--eval-min-new-tokens", type=int, default=None)
            parser.add_argument(
                "--log-eval-rollouts",
                action=argparse.BooleanOptionalAction,
                help="Log all responses for the first fixed prompt per RLVE evaluation environment to W&B Tables.",
            )

            return parser

        def add_algo_arguments(parser):
            parser.add_argument(
                "--ref-load",
                type=str,
                default=None,
                help=(
                    "The checkpoint for reference model. "
                    "When --load is not set, this will be used as the initial checkpoint for training. "
                ),
            )
            parser.add_argument(
                "--ref-ckpt-step", type=int, default=None, help="The checkpoint step for reference model. "
            )
            reset_arg(parser, "--load", type=str, default=None)
            reset_arg(parser, "--save", type=str, default=None)
            reset_arg(parser, "--save-interval", type=int, default=None)
            parser.add_argument("--transactional-checkpoints", action="store_true")
            reset_arg(parser, "--seed", type=int, default=1234)
            reset_arg(parser, "--clip-grad", type=float, default=1.0)
            reset_arg(parser, "--calculate-per-token-loss", action="store_true")

            parser.add_argument("--eps-clip", type=float, default=0.2, help="PPO clip range")
            parser.add_argument("--eps-clip-high", type=float, default=None, help="PPO clip upper range")
            parser.add_argument(
                "--eps-clip-c",
                type=float,
                default=None,
                help="lower bound of the value for Dual-clip PPO from https://arxiv.org/pdf/1912.09729",
            )
            parser.add_argument(
                "--kl-coef",
                type=float,
                default=0.00,
                help="KL penalty coefficient for reward shaping. This is applied to the reward signal before advantage calculation.",
            )
            parser.add_argument(
                "--loss-type",
                type=str,
                choices=["policy_loss", "sft_loss", "dpo_loss", "custom_loss"],
                default="policy_loss",
                help=(
                    "Choose loss type, currently support ppo policy_loss or sft_loss, "
                    "if custom_loss is set, we will use the function path from `--custom-loss-function-path`."
                ),
            )
            parser.add_argument(
                "--custom-loss-function-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom loss function, if the loss_type is `custom_loss`, "
                    "we will use this function to calculate the loss. "
                ),
            )
            parser.add_argument("--dpo-beta", type=float, default=0.1)
            parser.add_argument("--dpo-online", action="store_true", default=False)
            parser.add_argument("--dpo-base-hf", type=str, default=None)
            parser.add_argument(
                "--kl-loss-type",
                type=str,
                choices=["kl", "k2", "k3", "low_var_kl"],
                default="kl",
                help="Choose KL loss type: kl, k2, k3 low_var_kl",
            )
            parser.add_argument(
                "--advantage-estimator",
                type=str,
                choices=["grpo", "gspo", "reinforce_plus_plus", "reinforce_plus_plus_baseline"],
                default="grpo",
            )
            parser.add_argument(
                "--disable-compute-advantages-and-returns",
                action="store_false",
                dest="compute_advantages_and_returns",
                help=(
                    "Whether to disable computing advantages and returns. "
                    "If set, we will not compute the advantages and returns, "
                    "This is useful for sft or custom loss function."
                ),
            )
            parser.add_argument(
                "--use-kl-loss", action="store_true", default=False, help="whether to use KL loss from GRPO"
            )
            parser.add_argument(
                "--kl-loss-coef",
                type=float,
                default=0.0,
                help="KL penalty coefficient for the loss function. This is added to the final PPO loss.",
            )
            parser.add_argument("--entropy-coef", type=float, default=0.0, help="Entropy loss coef")
            parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor for rewards in REINFORCE++.")
            parser.add_argument("--normalize-advantages", action="store_true", default=False)
            parser.add_argument(
                "--disable-grpo-std-normalization",
                action="store_false",
                dest="grpo_std_normalization",
                help="from Dr.GRPO https://arxiv.org/pdf/2503.20783",
            )
            parser.add_argument(
                "--disable-rewards-normalization",
                action="store_false",
                dest="rewards_normalization",
                help="Disable rewards normalization",
            )
            parser.add_argument(
                "--use-rollout-entropy",
                action="store_true",
                default=False,
                help=(
                    "Whether to calculate the entropy when calculating the logprobs from actor and reference model. "
                    "This is useful for doing special loss mask."
                ),
            )
            # Off-Policy Correction using Importance Sampling: https://fengyao.notion.site/off-policy-rl
            parser.add_argument(
                "--use-tis",
                action="store_true",
                default=False,
                help="Enable TIS from https://fengyao.notion.site/off-policy-rl for off-policy importance sampling.",
            )
            parser.add_argument(
                "--tis-clip",
                type=float,
                default=2.0,
                help="Clipping threshold C for importance sampling ratios to control variance.",
            )
            parser.add_argument(
                "--tis-clip-low",
                type=float,
                default=0,
                help="Lower bound clipping threshold C for importance sampling ratios to control variance.",
            )
            return parser

        # wandb
        def add_wandb_arguments(parser):
            # wandb parameters
            parser.add_argument("--use-wandb", action="store_true", default=False)
            parser.add_argument(
                "--wandb-mode",
                type=str,
                default=None,
                choices=["online", "offline", "disabled"],
                help="W&B mode: online (default), offline (local only), or disabled. Overrides WANDB_MODE env var.",
            )
            parser.add_argument(
                "--wandb-dir",
                type=str,
                default=None,
                help="Directory to store wandb logs. Default is ./wandb in current directory.",
            )
            parser.add_argument("--wandb-key", type=str, default=None)
            parser.add_argument("--wandb-host", type=str, default=None)
            parser.add_argument("--wandb-team", type=str, default=None)
            parser.add_argument("--wandb-group", type=str, default=None)
            parser.add_argument(
                "--wandb-run-name",
                type=str,
                default=None,
                help="Name of the wandb run. Defaults to --wandb-group; set it when several runs share a group.",
            )
            reset_arg(parser, "--wandb-project", type=str, default=None)
            parser.add_argument(
                "--disable-wandb-random-suffix",
                action="store_false",
                dest="wandb_random_suffix",
                default=True,
                help=(
                    "Whether to add a random suffix to the wandb run name. "
                    "By default, we will add a random 6 length string with characters to the run name."
                ),
            )
            parser.add_argument(
                "--wandb-always-use-train-step",
                action="store_true",
                default=False,
                help=(
                    "Whether to always use train step as the step metric in wandb. "
                    "If set, we will always use the train steps for wandb logging, "
                    "otherwise, will use rollout step for most info other than train/*. "
                ),
            )
            parser.add_argument(
                "--log-multi-turn",
                action="store_true",
                default=False,
                help="Whether to log information for multi-turn rollout.",
            )
            parser.add_argument(
                "--log-passrate",
                action="store_true",
                default=False,
                help="Whether to turn on passrate logging, which will log the pass@n of the responses in the rollout.",
            )
            parser.add_argument("--wandb-run-id", type=str, default=None)
            return parser

        # debug
        def add_debug_arguments(parser):
            parser.add_argument(
                "--save-debug-rollout-data",
                type=str,
                default=None,
                help=(
                    "Save the rollout data to this path for debugging. "
                    "The file will be saved to `save_debug_rollout_data.format(rollout_id)`."
                ),
            )
            parser.add_argument(
                "--save-rollouts-jsonl",
                type=str,
                default=None,
                help="Write one HF-friendly JSONL file per training rollout; path may contain {rollout_id}.",
            )
            parser.add_argument(
                "--load-debug-rollout-data",
                type=str,
                default=None,
                help=(
                    "Load the rollout data from this path for debugging. "
                    "The file will be loaded from `load_debug_rollout_data.format(rollout_id)`. "
                    "When this is enabled, slime will not instantiate sglang servers."
                ),
            )
            parser.add_argument(
                "--debug-rollout-only",
                action="store_true",
                default=False,
                help=(
                    "Whether to only run the rollout generation without training. "
                    "This is useful for debugging the rollout generation function."
                ),
            )
            parser.add_argument(
                "--debug-train-only",
                action="store_true",
                default=False,
                help=(
                    "Whether to only run the training without sglang servers. "
                    "This is useful for debugging the rollout generation function."
                ),
            )
            parser.add_argument(
                "--save-debug-train-data",
                type=str,
                default=None,
                help=(
                    "Save the train data to this path for debugging. "
                    "The file will be saved to `save_debug_train_data.format(rollout_id)`."
                ),
            )
            parser.add_argument(
                "--dump-details",
                type=str,
                default=None,
                help=("Dump all details of training for post-hoc analysis and visualization."),
            )
            return parser

        def add_network_arguments(parser):
            parser.add_argument("--http-proxy", type=str, default=None)
            parser.add_argument("--use-http2", action="store_true", default=False)
            return parser

        def add_reward_model_arguments(parser):
            parser.add_argument(
                "--rm-type",
                type=str,
                default=None,
                help="Type of the reward model",
            )
            parser.add_argument(
                "--reward-key",
                type=str,
                default=None,
                help=(
                    "Some reward model may return a dict instead of a value, "
                    "this is the key to extract the reward value from the dict. "
                ),
            )
            parser.add_argument(
                "--eval-reward-key",
                type=str,
                default=None,
                help="The eval variant for --reward-key",
            )
            parser.add_argument(
                "--group-rm", action="store_true", default=False, help="Whether to do rm on a whole group."
            )
            parser.add_argument(
                "--rm-url",
                type=str,
                default=None,
                help="URL for the reward model service for --rm-type remote_rm, e.g. http://localhost:8000",
            )
            parser.add_argument(
                "--custom-rm-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom reward model function. "
                    "If set, we will use this function to calculate the reward instead of the default one. "
                    "The function should have the signature `def custom_rm(args, sample) -> float`."
                ),
            )
            parser.add_argument(
                "--custom-reward-post-process-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom function that will post process reward, by default it will be the normalization for grpo. "
                ),
            )
            return parser

        def add_rollout_buffer_arguments(parser):
            parser.add_argument(
                "--rollout-buffer-url",
                type=str,
                default=None,
                help="URL for the rollout buffer",
            )

            parser.add_argument(
                "--fetch-trajectory-retry-times",
                type=int,
                default=-1,
                help="Number of times to retry fetching trajectory, -1 means unlimited retry",
            )
            parser.add_argument(
                "--min-batch-collection-ratio",
                type=float,
                default=1,
                help="Minimum batch collection ratio",
            )
            parser.add_argument(
                "--rollout-task-type",
                type=str,
                default="math",
            )
            parser.add_argument(
                "--loss-mask-type",
                type=str,
                default="qwen",
                choices=["qwen", "distill_qwen"],
                help="Loss mask type",
            )
            return parser

        def add_custom_megatron_plugins_arguments(parser):
            """
            Add custom Megatron plugins arguments.
            This is a placeholder for any additional arguments that might be needed.
            """
            # Custom arguments can be added here
            parser.add_argument(
                "--custom-megatron-init-path",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--custom-megatron-before-log-prob-hook-path",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--custom-megatron-before-train-step-hook-path",
                type=str,
                default=None,
            )
            return parser

        def add_ci_arguments(parser):
            parser.add_argument(
                "--ci-test",
                action="store_true",
            )
            return parser

        def add_custom_arguments(parser) :
            parser.add_argument(
                "--rlve",
                action = "store_true",
                default = False,
                help = "Enable RLVE training.",
            )
            parser.add_argument(
                "--environment-list",
                type = str,
                nargs = '+',
                default = None,
                help = (
                    "List of verifiable environments to train on. "
                    "Accepts multiple string values (e.g. --environment-list env1 env2). "
                    "Default is None; when args.rlve is True, it should be provided."
                    "Accessible as args.environment_list; type is list of str if provided, else None."
                ),
            )
            parser.add_argument("--rlve-prompt-pool", type=str, default=None)
            parser.add_argument("--rlve-balanced-difficulty", action="store_true",
                                help="Cycle through every environment/difficulty pair once before repeating")
            parser.add_argument("--rlve-mix-file", type=str, default=None)
            parser.add_argument("--rlve-mix-index", type=int, default=None)
            parser.add_argument("--rlve-mix-seed", type=int, default=20260927)
            parser.add_argument("--rlve-final-ood-eval-path", type=str, default=None)
            parser.add_argument("--rlve-final-ood-only", action="store_true")
            parser.add_argument(
                "--initial-difficulty",
                type = int,
                default = 0,
                help = "Initial difficulty (upper bound) for each environment.",
            )
            parser.add_argument(
                "--difficulty-sliding-window-size",
                type = int,
                default = 4,
                help = "Size of the sliding window for problem difficulty (of each environment).",
            )
            parser.add_argument(
                "--min-metric-to-increase-difficulty",
                type = float,
                default = 0.9,
                help = "When the metric (for a specific environment) exceeds this value, the difficulty (for that environment) will be increased.",
            )
            parser.add_argument(
                "--min-prompts-before-difficulty-check",
                type = int,
                default = 8,
                help = "Minimum number of prompts before performing a difficulty check (checking the metric against args.min_metric_to_increase_difficulty).",
            )
            parser.add_argument(
                "--answer-marker-type",
                type = str,
                default = r"\boxed{}",
                help = "The type of answer marker to use.",
            )
            parser.add_argument(
                "--custom-prompt-preprocessor",
                type = str,
                required = True,
                choices = ("TinyZero", "ChatTemplate_NoSystemPrompt", "ChatTemplate_BoxedAnswer", ),
                help = "Choose a custom prompt preprocessor.",
            )
            return parser

        # Add custom arguments in front to prevent overwritten some slime arguments.
        if add_custom_arguments is not None:
            parser = add_custom_arguments(parser)

        parser = add_cluster_arguments(parser)
        parser = add_rollout_arguments(parser)
        parser = add_data_arguments(parser)
        parser = add_eval_arguments(parser)
        parser = add_algo_arguments(parser)
        parser = add_wandb_arguments(parser)
        parser = add_debug_arguments(parser)
        parser = add_sglang_arguments(parser)
        parser = add_network_arguments(parser)
        parser = add_reward_model_arguments(parser)
        parser = add_rollout_buffer_arguments(parser)
        parser = add_ci_arguments(parser)

        # For megatron
        parser = add_custom_megatron_plugins_arguments(parser)
        try:
            parser.add_argument("--padded-vocab-size", type=int, default=None)
        except:
            pass

        return parser

    return add_slime_arguments


def warning_for_unfinished_backend(backend: str):
    print("⚠️ " * 50)
    print(
        f"⚠️  SLIME_BACKEND {backend} is experimental and not yet verified.\n"
        "⚠️  Please avoid using it unless you are actively developing it."
    )
    print("⚠️ " * 50)


def parse_args(add_custom_arguments=None):
    add_slime_arguments = get_slime_extra_args_provider(add_custom_arguments)

    backend = os.environ.get("SLIME_BACKEND", "megatron").lower()
    if backend == "megatron":
        from slime.backends.megatron_utils import parse_args as megatron_parse_args
        from slime.backends.megatron_utils import set_default_megatron_args
        from slime.backends.megatron_utils import validate_args as megatron_validate_args

        args = megatron_parse_args(extra_args_provider=add_slime_arguments)
        if getattr(args, "hf_checkpoint", None):
            hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
            hf_validate_args(args, hf_config)

        args.rank = 0
        args.world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
        args = set_default_megatron_args(args)
    elif backend == "xtuner":
        warning_for_unfinished_backend(backend)
        from slime.backends.xtuner_utils.arguments import parse_args as xtuner_parse_args

        args = xtuner_parse_args(add_slime_arguments)
        args.rank = 0
        args.world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
    else:
        warning_for_unfinished_backend(backend)
        from slime.backends.fsdp_utils.arguments import load_fsdp_args

        args = load_fsdp_args(extra_args_provider=add_slime_arguments)

    if args.kl_coef != 0 or args.use_kl_loss:
        if not os.path.exists(args.ref_load):
            raise FileNotFoundError(f"ref_load {args.ref_load} does not exist, please check the path.")

        if not os.path.exists(os.path.join(args.ref_load, "latest_checkpointed_iteration.txt")):
            print(
                f"ref_load {args.ref_load} does not have latest_checkpointed_iteration.txt, "
                "please make sure it is a valid megatron checkpoint directory."
            )

    # TODO: During loading, we need to set the start_rollout_id here.
    if (
        args.load is None
        or not os.path.exists(args.load)
        or not os.path.exists(os.path.join(args.load, "latest_checkpointed_iteration.txt"))
    ):
        args.no_load_optim = True
        args.no_load_rng = True
        args.finetune = True
        args.load = args.ref_load
        if args.ref_ckpt_step is not None:
            args.ckpt_step = args.ref_ckpt_step
        args.start_rollout_id = 0

    if args.eval_interval is not None:
        assert args.eval_prompt_data is not None, "eval_prompt_data must be set when eval_interval is set"
        if len(args.eval_prompt_data) == 1:
            print(f"[legacy] only one eval_prompt_data detected, will assume it is data for aime")
            args.eval_prompt_data = ["aime", args.eval_prompt_data[0]]
        assert len(args.eval_prompt_data) % 2 == 0, "eval prompt data will need to be in pairs"

    if args.opd_teacher_dir is not None:
        assert args.opd_teacher_cache_size >= 0
        assert os.path.isdir(args.opd_teacher_dir), f"Missing OPD teacher directory: {args.opd_teacher_dir}"
        assert args.opd_teacher_gpus > 0 and args.colocate and args.actor_num_nodes == 1
        assert args.loss_type == "custom_loss" and not args.compute_advantages_and_returns
        assert args.rollout_temperature == 1.0, "OPD log probabilities require sampling temperature 1"
    else:
        assert args.opd_teacher_gpus == 0, "OPD teacher GPUs require --opd-teacher-dir"

    eos_pair = (getattr(args, "opd_student_eos_token_id", None), getattr(args, "opd_teacher_eos_token_id", None))
    if any(value is not None for value in eos_pair):
        if (any(value is None or value < 0 for value in eos_pair) or args.opd_teacher_dir is None):
            raise ValueError("OPD EOS alignment requires a teacher bank and both nonnegative stop-token IDs")
        if not set(eos_pair).issubset(set(args.rollout_stop_token_ids or [])):
            raise ValueError("OPD EOS alignment requires --rollout-stop-token-ids to include both EOS aliases")
        if args.opd_replay_fraction:
            raise ValueError("OPD EOS alignment currently supports fresh trajectories only; cached scores may use old labels")

    if not 0 <= args.opd_replay_fraction <= 0.5:
        raise ValueError("OPD replay fraction must be in [0, 0.5]")
    if args.opd_replay_capacity_per_env < 1 or not 0 < args.opd_replay_clip_epsilon < 1:
        raise ValueError("OPD replay requires a positive capacity and clipping epsilon in (0, 1)")
    if args.opd_replay_fraction:
        if (args.opd_teacher_dir is None or args.n_samples_per_prompt != 1
                or len(args.environment_list) != 1 or args.rollout_batch_size < 2
                or args.custom_loss_function_path != "slime.backends.megatron_utils.opd_loss.opd_loss_function"):
            raise ValueError("OPD replay requires single-environment student merges with the OPD loss and batch >=2")
        if args.opd_replay_load and not os.path.isfile(args.opd_replay_load):
            raise ValueError(f"Missing committed OPD replay snapshot: {args.opd_replay_load}")

    if args.fixed_rollout_policy:
        if backend != "megatron" or not args.colocate:
            raise ValueError("Fixed rollout policy requires the colocated Megatron backend")
        if args.keep_old_actor or args.debug_rollout_only or args.debug_train_only:
            raise ValueError("Fixed rollout policy is incompatible with old-actor and debug modes")
        if args.loss_type == "dpo_loss" or args.advantage_estimator != "grpo":
            raise ValueError("Fixed rollout policy currently requires the GRPO objective")
        if not args.use_tis:
            raise ValueError("Fixed rollout policy requires --use-tis for off-policy correction")

    if args.loss_type == "dpo_loss":
        assert args.dpo_beta > 0 and not args.compute_advantages_and_returns
        assert args.n_samples_per_prompt == 2 and args.micro_batch_size == 2
        assert not args.use_dynamic_batch_size and args.balance_data
        assert args.rollout_temperature == 1.0
        if args.dpo_online:
            assert args.dpo_base_hf and os.path.isdir(args.dpo_base_hf)
            assert args.load_debug_rollout_data is None
    else:
        assert not args.dpo_online, "Online DPO requires the DPO loss"

    if args.early_stop_val_accuracy is not None:
        assert 0 < args.early_stop_val_accuracy <= 1
        assert args.eval_interval is not None and (args.rlve or args.rollout_global_dataset)
        assert args.save is None or args.transactional_checkpoints or os.environ.get("RLVE_FINAL_WEIGHTS_ONLY") == "1"
        eval_names = set(args.eval_prompt_data[::2])
        assert all(name.split("/", 1)[0] in eval_names for name in args.early_stop_eval_names)
    if args.early_stop_patience_evals is not None:
        assert args.early_stop_patience_evals > 0
        assert args.early_stop_min_rollouts >= 0
        assert 0 <= args.early_stop_min_delta < 1
        assert args.eval_interval is not None and args.save is not None
        assert args.transactional_checkpoints and args.rlve
    if args.require_early_stop:
        assert args.early_stop_patience_evals is not None or args.early_stop_val_accuracy is not None

    if args.save_interval is not None:
        assert args.save is not None, "'--save' is required when save_interval is set."

    assert not (args.kl_coef != 0 and args.kl_loss_coef != 0), "Only one of kl_coef and kl_loss_coef can be set"

    if args.advantage_estimator in ["reinforce_plus_plus", "reinforce_plus_plus_baseline"]:
        assert args.normalize_advantages, (
            "The 'reinforce_plus_plus' and 'reinforce_plus_plus_baseline' advantage estimators "
            "require advantage normalization. Please add `--normalize-advantages` to your command."
        )

    if args.use_dynamic_batch_size:
        assert args.max_tokens_per_gpu is not None, "max_tokens_per_gpu must be set when use_dynamic_batch_size is set"
        if args.log_probs_max_tokens_per_gpu is None:
            args.log_probs_max_tokens_per_gpu = args.max_tokens_per_gpu

    assert args.answer_forcing_extra_tokens > 0, "answer_forcing_extra_tokens must be positive"

    if args.eps_clip_high is None:
        args.eps_clip_high = args.eps_clip

    if args.eval_reward_key is None:
        args.eval_reward_key = args.reward_key

    if args.dump_details is not None:
        args.save_debug_rollout_data = f"{args.dump_details}/rollout_data/{{rollout_id}}.pt"
        args.save_debug_train_data = f"{args.dump_details}/train_data/{{rollout_id}}_{{rank}}.pt"

    if args.load_debug_rollout_data is not None:
        print(
            f"load_debug_rollout_data {args.load_debug_rollout_data} is set, "
            "loading replay batches."
        )
        # DPO replays offline pairs but still evaluates the changing student.
        if args.loss_type != "dpo_loss":
            args.debug_train_only = True

    if args.debug_rollout_only:
        if args.colocate and args.rollout_num_gpus is None:
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
        else:
            args.actor_num_gpus_per_node = min(8, args.rollout_num_gpus)
            args.actor_num_nodes = args.rollout_num_gpus // args.actor_num_gpus_per_node
        args.colocate = False
        args.offload = False

    assert not (args.debug_rollout_only and args.debug_train_only), (
        "debug_rollout_only and debug_train_only cannot be set at the same time, " "please set only one of them."
    )

    # always true on offload for colocate at the moment.
    if args.colocate:
        args.offload = True
        if args.rollout_num_gpus != args.actor_num_gpus_per_node * args.actor_num_nodes:
            print(
                f"rollout_num_gpus {args.rollout_num_gpus} != actor_num_gpus_per_node {args.actor_num_gpus_per_node} "
                f"* actor_num_nodes {args.actor_num_nodes}, overriding rollout_num_gpus to match actor_num_gpus_per_node * actor_num_nodes."
            )
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes

    if args.eval_function_path is None:
        args.eval_function_path = args.rollout_function_path

    if args.num_steps_per_rollout is not None:
        global_batch_size = args.rollout_batch_size * args.n_samples_per_prompt // args.num_steps_per_rollout
        if args.global_batch_size is not None:
            assert args.global_batch_size == global_batch_size, (
                f"global_batch_size {args.global_batch_size} is not equal to "
                f"rollout_batch_size {args.rollout_batch_size} * n_samples_per_prompt {args.n_samples_per_prompt} "
                f"// num_steps_per_rollout {args.num_steps_per_rollout}"
            )
        args.global_batch_size = global_batch_size

    assert args.rollout_batch_size * args.n_samples_per_prompt % args.global_batch_size == 0, (
        f"rollout_batch_size {args.rollout_batch_size} * n_samples_per_prompt {args.n_samples_per_prompt} "
        f"is not a multiple of global_batch_size {args.global_batch_size}"
    )
    assert args.train_reuse_epochs >= 1, "train_reuse_epochs must be positive"
    if args.rollout_anneal_thresholds is not None or args.rollout_anneal_sizes is not None:
        assert args.rollout_anneal_thresholds and args.rollout_anneal_sizes
        assert len(args.rollout_anneal_thresholds) == len(args.rollout_anneal_sizes)
        assert all(0 <= x <= 1 for x in args.rollout_anneal_thresholds)
        assert args.rollout_anneal_thresholds == sorted(set(args.rollout_anneal_thresholds))
        assert all(0 < x < args.n_samples_per_prompt for x in args.rollout_anneal_sizes)
        assert args.rollout_anneal_sizes == sorted(args.rollout_anneal_sizes, reverse=True)
        assert args.num_steps_per_rollout == 1 and args.train_reuse_epochs == 1
        assert args.eval_interval is not None
    if args.dynamic_sampling_max_rejections is not None:
        assert args.dynamic_sampling_filter_path is not None
        assert args.dynamic_sampling_max_rejections >= 0

    if args.n_samples_per_prompt == 1:
        args.grpo_std_normalization = False
        print("n_samples_per_prompt is set to 1, grpo_std_normalization will be set to False.")

    if args.over_sampling_batch_size is None:
        args.over_sampling_batch_size = args.rollout_batch_size

    assert args.over_sampling_batch_size >= args.rollout_batch_size, (
        f"over_sampling_batch_size {args.over_sampling_batch_size} should be greater than or equal to "
        f"rollout_batch_size {args.rollout_batch_size}"
    )

    if args.num_epoch is not None:
        if args.num_rollout is not None:
            print("Both num_epoch and num_rollout are set, num_epoch will be ignored.")
        else:
            assert args.rollout_global_dataset, (
                "num_epoch is set, but rollout_global_dataset is not set, "
                "please remove --disable-rollout-global-dataset to use num_epoch"
            )
    else:
        # if num_epoch is not set, we should set num_rollout
        assert args.num_rollout is not None, (
            "num_epoch is not set, but num_rollout is not set, " "please set --num-rollout or --num-epoch"
        )

    if backend == "megatron":
        megatron_validate_args(args)

        # always use varlen
        args.variable_seq_lengths = True
        if getattr(args, "moe_token_dispatcher_type", None) == "allgather":
            print(
                "--moe-token-dispatcher-type allgather does not support variable sequence length, "
                "please use alltoall dispatcher instead."
            )
            args.moe_token_dispatcher_type = "alltoall"

    if any(value is not None for value in (args.rlve_prompt_pool, args.rlve_mix_file, args.rlve_mix_index)):
        if not args.rlve or not all(value is not None for value in (args.rlve_prompt_pool, args.rlve_mix_file, args.rlve_mix_index)):
            raise ValueError("frozen mixing requires --rlve, --rlve-prompt-pool, --rlve-mix-file, and --rlve-mix-index")
        if args.initial_difficulty != 0 or args.min_metric_to_increase_difficulty <= 1:
            raise ValueError("frozen mixing requires difficulty fixed at zero")
        from slime.utils.rlve_mixing import load_candidate

        manifest, candidate = load_candidate(
            args.rlve_mix_file, args.rlve_mix_index, args.environment_list, args.rlve_prompt_pool
        )
        args.rlve_mix_weights = candidate["weights"]
        args.rlve_mix_alpha = candidate["alpha"]
        args.rlve_pool_sha256 = manifest["pool_sha256"]

    if args.rlve_balanced_difficulty:
        if (not args.rlve or not args.environment_list or args.rlve_prompt_pool is not None or
                args.initial_difficulty < 0 or
                args.difficulty_sliding_window_size < args.initial_difficulty + 1 or
                args.min_metric_to_increase_difficulty <= 1):
            raise ValueError("balanced difficulty requires native RLVE sampling, fixed levels 0..initial_difficulty, and a full difficulty window")

    if args.rlve_final_ood_only and args.rlve_final_ood_eval_path is None:
        raise ValueError("final-only OOD evaluation requires --rlve-final-ood-eval-path")
    if args.rlve_final_ood_eval_path is not None:
        from pathlib import Path

        if not args.rlve or args.eval_interval is None or args.num_rollout < 21:
            raise ValueError("final OOD eval requires RLVE and at least 21 training updates")
        if not Path(args.rlve_final_ood_eval_path).is_file():
            raise ValueError(f"missing final OOD eval: {args.rlve_final_ood_eval_path}")
        pairs = list(zip(args.eval_prompt_data[::2], args.eval_prompt_data[1::2]))
        if not any(name == "test_envs" for name, _ in pairs):
            raise ValueError("final OOD eval requires a test_envs evaluation dataset")

    sglang_validate_args(args)

    return args


def hf_validate_args(args, hf_config):
    equal = lambda x, y: x == y
    for hf_config_name, megatron_config_name, compare_fn in [
        ("hidden_size", "hidden_size", equal),
        ("num_attention_heads", "num_attention_heads", equal),
        ("num_hidden_layers", "num_layers", equal),
        ("intermediate_size", "ffn_hidden_size", equal),
        ("tie_word_embeddings", "untie_embeddings_and_output_weights", lambda x, y: not x == y),
        ("rms_norm_eps", "norm_epsilon", equal),
        ("rope_theta", "rotary_base", equal),
    ]:
        if hasattr(hf_config, hf_config_name):
            assert compare_fn(getattr(hf_config, hf_config_name), getattr(args, megatron_config_name)), (
                f"{hf_config_name} in hf config {getattr(hf_config, hf_config_name)} is not equal to "
                f"{megatron_config_name} {getattr(args, megatron_config_name)}, please check the config."
            )
