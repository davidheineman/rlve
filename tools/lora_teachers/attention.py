"""Select the pinned image's FlashAttention 2 implementation for Megatron."""

import logging


def configure_flash_attention_2(_args):
    # Miles invokes this hook before building the trainer model. TE otherwise
    # prefers FA4 on SM8x even though the pinned FA4 backward only supports SM9+.
    from transformer_engine.pytorch.attention.dot_product_attention.backends import (
        fa_utils,
    )

    if not fa_utils.is_installed or fa_utils.version.major != 2:
        raise RuntimeError("This run requires the pinned image's FlashAttention 2 backend")
    fa_utils.v3_is_installed = False
    fa_utils.v4_is_installed = False
    logging.getLogger(__name__).info("Selected FlashAttention %s for trainer forward and backward", fa_utils.version)
