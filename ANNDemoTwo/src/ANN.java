import java.util.Random;

public class ANN {
	// a method to output a String to represent a given array
	public String arrToString(double[] arr) {
		// initialize the string
		String arrayString = "";
		// add to "arrayString" for every element in arr[]
		for (int i = 0; i < arr.length; i++) {
			arrayString = arrayString + " [" + arr[i] + "],";
		}
		// return "arrayString"
		return arrayString;
	}

	// initializing the layers
	int inputLayerSize, hiddenLayerOneSize, hiddenLayerTwoSize, outputLayerSize;
	// initializing the biases
	double[] biasesLayerOne, biasesLayerTwo, biasesOutputLayer;
	// initializing the weights
	double[][] weightsAfterInputLayer, weightsAfterLayerOne, weightsAfterLayerTwo;
	// initializing the nodes
	double[] hiddenLayerOneNodes, hiddenLayerTwoNodes;

	// a boolean to keep track of if we normalized the data or not
	boolean normalized = false;

	// variables to keep track of normalization and denormalization
	double minInput, maxInput, minOutput, maxOutput;

	// private methods for changing the layers
	private void changeInputLayerSize(int newInputLayerSize) {
		if (newInputLayerSize <= 0) {
			throw new ArithmeticException("layers must have more than one node");
		} else {
			inputLayerSize = newInputLayerSize;
		}
	}

	private void changeHiddenLayerOneSize(int newHiddenLayerOneSize) {
		if (newHiddenLayerOneSize <= 0) {
			throw new ArithmeticException("layers must have more than one node");
		} else {
			hiddenLayerOneSize = newHiddenLayerOneSize;
		}
	}

	private void changeHiddenLayerTwoSize(int newHiddenLayerTwoSize) {
		if (newHiddenLayerTwoSize <= 0) {
			throw new ArithmeticException("layers must have more than one node");
		} else {
			hiddenLayerTwoSize = newHiddenLayerTwoSize;
		}
	}

	private void changeOutputLayerSize(int newOutputLayerSize) {
		if (newOutputLayerSize <= 0) {
			throw new ArithmeticException("layers must have more than one node");
		} else {
			outputLayerSize = newOutputLayerSize;
		}
	}

	// private methods to change the minInput, maxInput, minOutput, and maxOutput
	private void changeInputParameters(double min, double max) {
		// if minInput > maxInput (min should always be less than max)
		if (min > max) {
			throw new ArithmeticException("minInput must always be less than maxInput");
		} else {
			minInput = min;
			maxInput = max;
		}
	}

	private void changeOutputParameters(double min, double max) {
		// if minOutput > maxOutput (min should always be less than max)
		if (min > max) {
			throw new ArithmeticException("minInput must always be less than maxInput");
		} else {
			minOutput = min;
			maxOutput = max;
		}
	}

	// the instance of the Random class we will be using to initialize the ANN
	Random rand = new Random();

	// to randomize the values in an array between 0 and 1 (used for the biases
	// arrays)
	private void randomizeOneDimensionalArr(double[] arr) {
		for (int i = 0; i < arr.length; i++) {
			// generates a double on the interval [0.0, 1.0]
			arr[i] = rand.nextDouble();
		}
	}

	private void randomizeTwoDimensionalArr(double[][] arr) {
		for (int i = 0; i < arr.length; i++) {
			for (int j = 0; j < arr[i].length; j++) {
				arr[i][j] = rand.nextDouble();
			}
		}
	}

	public ANN(int inputLayerSize, int hiddenLayerOneSize, int hiddenLayerTwoSize, int outputLayerSize, double minInput,
			double maxInput, double minOutput, double maxOutput) {
		// calls all the methods for changing the size of the layers. They will throw
		// exceptions if illegal values are attempted
		changeInputLayerSize(inputLayerSize);
		changeHiddenLayerOneSize(hiddenLayerOneSize);
		changeHiddenLayerTwoSize(hiddenLayerTwoSize);
		changeOutputLayerSize(outputLayerSize);

		// initializing and randomizing all the biases (they were already initialized as
		// variables above, but now we are assigning them a length and thus a slot in
		// memory)
		biasesLayerOne = new double[hiddenLayerOneSize];
		randomizeOneDimensionalArr(biasesLayerOne);
		biasesLayerTwo = new double[hiddenLayerTwoSize];
		randomizeOneDimensionalArr(biasesLayerTwo);
		biasesOutputLayer = new double[outputLayerSize];
		randomizeOneDimensionalArr(biasesOutputLayer);

		// initializing and randomizing all the weights (they were already initialized
		// as variables above, but now we are assigning them a length and thus a slot in
		// memory)
		weightsAfterInputLayer = new double[inputLayerSize][hiddenLayerOneSize];
		randomizeTwoDimensionalArr(weightsAfterInputLayer);
		weightsAfterLayerOne = new double[hiddenLayerOneSize][hiddenLayerTwoSize];
		randomizeTwoDimensionalArr(weightsAfterLayerOne);
		weightsAfterLayerTwo = new double[hiddenLayerTwoSize][outputLayerSize];
		randomizeTwoDimensionalArr(weightsAfterLayerTwo);

		/*
		 * initializing all the nodes the nodes are the parts of each layer that will
		 * hold values as the calculation is done (they were already initialized as
		 * variables above, but now we are assigning them a length and thus a slot in
		 * memory)
		 * 
		 * the input layer nodes will come from the user input and the output layer will
		 * be a result of the calculation (so a blank output layer will be passed in by
		 * the user and the actual outputs will be returned)
		 */
		hiddenLayerOneNodes = new double[hiddenLayerOneSize];
		hiddenLayerTwoNodes = new double[hiddenLayerTwoSize];

		// changes the min and max for normalization and denormalization
		changeInputParameters(minInput, maxInput);
		changeOutputParameters(minOutput, maxOutput);
	}

	// the sigmoid method
	private double sigmoid(double x) {
		// Math.exp(x) is equivalent to (e^x)
		return (1 / (1 + (1 / Math.exp(x))));
	}

	// the method that will calculate a single layer of node values
	private void layerCalculate(double[] inputs, double[] targets, double[][] weights, double[] biases) {
		for (int i = 0; i < targets.length; i++) {
			// this will be the sum of all the inputs times all the weights plus a bias
			double sum = 0;
			for (int j = 0; j < inputs.length; j++) {
				sum += (inputs[j] * weights[j][i]);
			}
			// add the bias to "sum"
			sum += biases[i];
			// reasign the "sum" variable as the sigmoid of "sum"
			sum = sigmoid(sum);
			// assign the output to the target array after passing "sum" through the sigmoid
			// method
			targets[i] = sum;
		}
	}
	
	// the method that will normalize the input array
	private void normalize(double[] arr) {
		// do the normalization on each element of arr[]
		for (int i = 0; i < arr.length; i++) {
			arr[i] = ((arr[i] - minInput) / (maxInput - minInput));
		}
	}

	// the method that will denormalize the output array
	private void denormalize(double[] arr) {
		// do the denormalization on each element of arr[]
		for (int i = 0; i < arr.length; i++) {
			arr[i] = (((maxOutput - minOutput) * arr[i]) + minOutput);
		}
	}

	/*
	 * this is the method that will do the actual calculation of the ANN
	 * 
	 * it takes in an array of inputs and a blank array for the outputs it will
	 * change the values in the blank array for the outputs
	 */
	public void calculate(double[] inputs, double[] outputs) {
		if ((inputs.length != inputLayerSize) || (outputs.length != outputLayerSize)) {
			throw new ArithmeticException("You have not given the correct number of inputs and/or outputs");
		} else {
			// normalize the inputs
			normalize(inputs);

			// this will change all the values in "hiddenLayerOneNodes" to the calculated
			// values
			layerCalculate(inputs, hiddenLayerOneNodes, weightsAfterInputLayer, biasesLayerOne);
			// this will change all the values in "hiddenLayerTwoNodes" to the calculated
			// values
			layerCalculate(hiddenLayerOneNodes, hiddenLayerTwoNodes, weightsAfterLayerOne, biasesLayerTwo);
			// this will change all the values in "outputs" to the calculated values
			layerCalculate(hiddenLayerTwoNodes, outputs, weightsAfterLayerTwo, biasesOutputLayer);
			// by this part in the code, the outputs[] array should have been changed to the
			// desired outputs (but still in normalized form)

			// denormalize the outputs
			denormalize(outputs);

			System.out.println("=====================================================");
			System.out.println("the inputs array " + arrToString(inputs));
			System.out.println("the outputs array after calculation" + arrToString(outputs));
			System.out.println(hiddenLayerOneNodes[3]);
			System.out.println("=====================================================");
		}
	}
}