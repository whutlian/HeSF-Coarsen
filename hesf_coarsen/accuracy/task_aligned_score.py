from __future__ import annotations


def accuracy_first_score(
    *,
    delta_spec: float = 0.0,
    delta_feat: float = 0.0,
    delta_anchor_support: float = 0.0,
    delta_target_context: float = 0.0,
    delta_meta_recon: float = 0.0,
    delta_teacher_support: float = 0.0,
    lambda_spec: float = 0.25,
    lambda_feat: float = 0.1,
    lambda_anchor: float = 0.5,
    lambda_ctx: float = 0.25,
    lambda_meta: float = 0.25,
    lambda_teacher: float = 0.1,
) -> dict[str, float]:
    terms = {
        "delta_spec": float(delta_spec),
        "delta_feat": float(delta_feat),
        "delta_anchor_support": float(delta_anchor_support),
        "delta_target_context": float(delta_target_context),
        "delta_meta_recon": float(delta_meta_recon),
        "delta_teacher_support": float(delta_teacher_support),
    }
    score = (
        float(lambda_spec) * terms["delta_spec"]
        + float(lambda_feat) * terms["delta_feat"]
        + float(lambda_anchor) * terms["delta_anchor_support"]
        + float(lambda_ctx) * terms["delta_target_context"]
        + float(lambda_meta) * terms["delta_meta_recon"]
        + float(lambda_teacher) * terms["delta_teacher_support"]
    )
    return {**terms, "score_acc": float(score)}
