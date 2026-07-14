import pytest

import DECIMER.decimer as decimer_module
from DECIMER import predict_SMILES
from DECIMER import predict_SMILES_batch


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_imagetosmiles():
    img_path = "tests/caffeine.png"
    expected_result = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
    actual_result = predict_SMILES(img_path)
    assert expected_result == actual_result


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_imagetosmilesWithConfidence():
    img_path = "tests/caffeine.png"
    actual_result = predict_SMILES(img_path, confidence=True)

    for element, confidence in actual_result[1]:
        assert (
            confidence >= 0.9
        ), f"Confidence for element '{element}' is below 0.9 (confidence: {confidence})"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_imagetosmileshanddrawn():
    img_path = "tests/caffeine.png"
    expected_result = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
    actual_result = predict_SMILES(img_path, hand_drawn=True)
    assert expected_result == actual_result


def test_batch_prediction_uses_input_order(monkeypatch):
    image_paths = ["image_1.png", "image_2.png"]
    calls = []

    def mock_predict_smiles(image_input, confidence=False, hand_drawn=False):
        calls.append((image_input, confidence, hand_drawn))
        return "SMILES:{}".format(image_input)

    monkeypatch.setattr(decimer_module, "predict_SMILES", mock_predict_smiles)

    actual_result = predict_SMILES_batch(image_paths, confidence=True, hand_drawn=True)

    assert actual_result == ["SMILES:image_1.png", "SMILES:image_2.png"]
    assert calls == [
        ("image_1.png", True, True),
        ("image_2.png", True, True),
    ]
