Learn::Result Learn::_calculateIndividual(std::shared_ptr<TrainingExample> ex) const
{
	// ======== forward calculation
	// size of in should be correct
	if (ex->_input.size() != _numInputNodes) {
		throw std::invalid_argument("input has wrong number of nodes");
	}

	if (ex->_expectedOutput.size() != _numOutputNodes) {
		throw std::invalid_argument("output has wrong number of nodes");
	}

	// create layers
	std::vector<LinAlg::Vector> layers;
	_initializeLikeLayers(layers, ex->_input);

	// untransformed layers
	std::vector<LinAlg::Vector> layersLinear;
	_initializeLikeLayers(layersLinear, ex->_input);

	for (size_t i = 1; i < _numLayers; ++i) { // begin at 1 since 0 is the input layer
		// std::cout << "==== NEW LAYER LAYER " << i << std::endl;
		// std::cout << "THE INPUT LAYER IS\n" << layers[i - 1] << std::endl << std::endl;
		// std::cout << "THE WEIGHTS ARE\n" << _weights[i] << std::endl << std::endl;
		// std::cout << "THE BIASES ARE\n" << _biases[i] << std::endl << std::endl;

		layersLinear[i] = (LinAlg::Vector)LinAlg::mmul(_weights[i], layers[i - 1]) + _biases[i];

		if (i == _numLayers - 1)
			layers[i] = _scalingFuncOutput(layersLinear[i]);
		else
			layers[i] = _scalingFunc(layersLinear[i]);

		// std::cout << "THE LINEAR RESULT IS\n" << layersLinear[i] << std::endl << std::endl;
		// std::cout << "THE SCALED RESULT IS\n" << layers[i] << std::endl << std::endl;
	}

	LinAlg::Vector outputLayer = layers.back(); // make sure to copy, as will be moved into the Result struct
	const LinAlg::Vector outputLayerLinear = layersLinear.back();

	double cost = _costFunc(outputLayer, ex->_expectedOutput);

	// ======== backward calculation
	std::vector<LinAlg::Vector> errorTerms;
	_initializeLikeLayers(errorTerms, {});
	// first calculate final layer's error terms
	for (size_t i = 0; i < _numOutputNodes; ++i) {
		errorTerms[_numLayers - 1][i] =
			_dCostFunc(outputLayer[i], ex->_expectedOutput[i]) * _dScalingFuncOutput(outputLayerLinear[i]);
	}

	// then calculate all hidden layers' error terms
	for (size_t k = _numHiddenLayers; k > 0; --k) {
		for (size_t i = 0; i < _numHiddenLayerNodes; ++i) {
			double weightedErrors = 0;
			for (size_t j = 0; j < errorTerms[k + 1].size(); ++j) {
				weightedErrors += errorTerms[k + 1][j] * _weights[k + 1][j][i];
			}

			errorTerms[k][i] = weightedErrors * _dScalingFunc(layersLinear[k][i]);
		}
	}
	// now get weight gradient
	std::vector<LinAlg::Matrix> weightGradient;
	_initializeLikeWeights(weightGradient);

	for (size_t k = 1; k < _numLayers; ++k) {
		for (size_t i = 0; i < weightGradient[k].size().first; ++i) {
			for (size_t j = 0; j < weightGradient[k].size().second; ++j) {
				weightGradient[k][i][j] = errorTerms[k][i] * layers[k - 1][j]; // this takes lots of time
			}
		}
	}
	// now get bias gradient
	std::vector<LinAlg::Vector> biasGradient;
	_initializeLikeBiases(biasGradient);

	for (size_t k = 1; k < _numLayers; ++k) {
		for (size_t i = 0; i < errorTerms[k].size(); ++i) {
			biasGradient[k][i] = errorTerms[k][i];
		}
	} // the biases are basically the same as the error terms

	//
	size_t maxCalculatedIndex = 0;
	double maxCalculated = 0;
	size_t maxExpectedIndex = 0;
	double maxExpected = 0;
	for (size_t i = 0; i < outputLayer.size(); ++i) {
		if (outputLayer[i] > maxCalculated) {
			maxCalculated = outputLayer[i];
			maxCalculatedIndex = i;
		}
	}

	for (size_t i = 0; i < ex->_expectedOutput.size(); ++i) {
		if (ex->_expectedOutput[i] > maxExpected) {
			maxExpected = ex->_expectedOutput[i];
			maxExpectedIndex = i;
		}
	}

	// std::cout << "expected " << maxExpectedIndex << " and got " << maxCalculatedIndex << std::endl;

	double accuracy = (maxExpectedIndex == maxCalculatedIndex) ? 1.0 : 0.0;
	return Result(std::move(outputLayer), std::move(cost), std::move(weightGradient), std::move(biasGradient),
				  accuracy);
}