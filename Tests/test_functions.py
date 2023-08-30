import pytest
from DECIMER import predict_SMILES, predict_SMILES_with_confidence


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_imagetosmiles():
    img_path = "Tests/caffeine.png"
    expected_result = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
    actual_result = predict_SMILES(img_path)
    assert expected_result == actual_result

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_imagetosmilesWithConfidence():
    img_path = "Tests/caffeine.png"
    expected_result = "[('C', 0.9889417), ('N', 0.9882632), ('1', 0.996114), ('C', 0.9914259), ('=', 0.99824154), ('N', 0.9983388), ('C', 0.9972761), ('2', 0.99852365), ('=', 0.9978861), ('C', 0.99654514), ('1', 0.99783903), ('C', 0.99765694), ('(', 0.9990073), ('=', 0.9916898), ('O', 0.9985745), (')', 0.99965775), ('N', 0.99692804), ('(', 0.99895144), ('C', 0.9972687), (')', 0.9992637), ('C', 0.99937975), ('(', 0.9995913), ('=', 0.9974962), ('O', 0.9977519), (')', 0.9994863), ('N', 0.9979286), ('2', 0.9964923), ('C', 0.9951383)]"
    actual_result = predict_SMILES_with_confidence(img_path)
    assert expected_result == actual_result