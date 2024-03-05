import pytest

from DECIMER import predict_SMILES
from DECIMER import predict_SMILES_with_confidence


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_imagetosmiles():
    img_path = "Tests/caffeine.png"
    expected_result = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
    actual_result = predict_SMILES(img_path)
    assert expected_result == actual_result


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_imagetosmilesWithConfidence():
    img_path = "Tests/caffeine.png"
    actual_result = predict_SMILES_with_confidence(img_path)

    for element, confidence in actual_result:
        assert (
            confidence >= 0.9
        ), f"Confidence for element '{element}' is below 0.9 (confidence: {confidence})"
