from pathlib import Path


def test_required_files():

    assert Path("models/best_model.pkl").exists()

    assert Path("artifacts/scaler.pkl").exists()

    assert Path("artifacts/model_metadata.json").exists()