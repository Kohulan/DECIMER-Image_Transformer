import pytest

from DECIMER import predict_SMILES


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
