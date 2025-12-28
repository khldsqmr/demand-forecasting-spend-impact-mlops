"""
Kubeflow Pipeline: Demand Forecasting & Spend Impact
---------------------------------------------------

This pipeline orchestrates the full ML workflow:
1. Feature engineering
2. Model training
3. Prediction generation
4. Financial impact analysis

All steps wrap existing, tested Python scripts.
"""

from kfp import dsl
from kfp.dsl import component


# ============================================================
# Pipeline Components
# ============================================================

@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "numpy", "scikit-learn", "joblib"]
)
def feature_engineering_op():
    """Run feature engineering"""
    import subprocess
    print("🚀 Running feature engineering step")
    subprocess.run(
        ["python", "src/features/feature_engineering.py"],
        check=True
    )
    print("✅ Feature engineering completed")


@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "numpy", "scikit-learn", "joblib"]
)
def train_baseline_model_op():
    """Train final baseline model"""
    import subprocess
    print("🚀 Training final baseline model")
    subprocess.run(
        ["python", "src/models/train_final_baseline.py"],
        check=True
    )
    print("✅ Model training completed")


@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "numpy", "scikit-learn", "joblib"]
)
def generate_predictions_op():
    """Generate baseline predictions"""
    import subprocess
    print("🚀 Generating baseline predictions")
    subprocess.run(
        ["python", "src/models/generate_baseline_predictions.py"],
        check=True
    )
    print("✅ Predictions generated")


@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "numpy"]
)
def financial_impact_op():
    """Run financial impact analysis"""
    import subprocess
    print("🚀 Running financial impact analysis")
    subprocess.run(
        ["python", "src/evaluation/spend_impact.py"],
        check=True
    )
    print("✅ Financial impact analysis completed")


# ============================================================
# Pipeline Definition
# ============================================================

@dsl.pipeline(
    name="Demand Forecasting & Spend Impact Pipeline",
    description="End-to-end ML pipeline for demand forecasting and financial impact analysis"
)
def demand_forecasting_pipeline():

    feature_step = feature_engineering_op()

    train_step = train_baseline_model_op().after(feature_step)

    predict_step = generate_predictions_op().after(train_step)

    impact_step = financial_impact_op().after(predict_step)


# ============================================================
# Compilation Entrypoint
# ============================================================

if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        pipeline_func=demand_forecasting_pipeline,
        package_path="kubeflow/demand_forecasting_pipeline.yaml"
    )

    print("📦 Kubeflow pipeline compiled successfully")
