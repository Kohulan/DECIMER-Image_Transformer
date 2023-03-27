import pytest
from DECIMER import predict_SMILES


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_imagetosmiles():
    img_path = "Tests/caffeine.png"
    expected_result = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    actual_result = predict_SMILES(img_path)
    assert expected_result == actual_result
