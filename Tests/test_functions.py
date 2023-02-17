import pytest
from DECIMER import predict_SMILES

def get_imagetosmiles():
	img_path = 'Tests/caffeine.png'
	expected_result = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
	actual_result = predict_SMILES(image_path)
	assert expected_result == actual_result