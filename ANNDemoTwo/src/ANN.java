import java.util.Random;
import java.io.*;
import java.util.Scanner;

public class ANN {
	/*
	 * =============================================================================
	 * ======= General Methods, Definitions, and Initializations for the ANN =======
	 * =============================================================================
	 */

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
	double min, max;

	// variables to keep track of datasets of inputs and outputs for training
//	double[] givenInputs, desiredOutputs;

	/*
	 * a variable to keep track of the learning weight
	 * 
	 * this variable will be set to 0.1 by default, but can be changed if the client
	 * wishes
	 */
	double learningRate = 0.1;

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

	// a private method to change the min and max
	private void changeParameters(double min, double max) {
		// if min > max (min should always be less than max)
		if (min > max) {
			throw new ArithmeticException("min must always be less than max");
		} else {
			this.min = min;
			this.max = max;
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

			/*
			 * turns the double into a negative 50% of the time
			 * 
			 * so now the double will be somewhere on the interval [-1.0, 1.0] instead of
			 * the interval [0.0, 1.0]
			 */
			if (rand.nextBoolean()) {
				arr[i] *= -1;
			}
		}
	}

	private void randomizeTwoDimensionalArr(double[][] arr) {
		for (int i = 0; i < arr.length; i++) {
			for (int j = 0; j < arr[i].length; j++) {
				arr[i][j] = rand.nextDouble();
				/*
				 * turns the double into a negative 50% of the time
				 * 
				 * so now the double will be somewhere on the interval [-1.0, 1.0] instead of
				 * the interval [0.0, 1.0]
				 */
				if (rand.nextBoolean()) {
					arr[i][j] *= -1;
				}
			}
		}
	}

	// a method for opening up a file
	private Scanner fileOpener(String file) {
		try {
			File f = new File(file);
			return new Scanner(f);
		} catch (FileNotFoundException e) {
			System.out.println("an error occurred trying to find the file: " + file);
			e.printStackTrace();
			return null;
		}
	}

	// a method for writing to a file
	private PrintStream fileWriter(String file) {
		try {
			File f = new File(file);
			System.out.println(file);
			return new PrintStream(f);
		} catch (IOException e) {
			System.out.println("an error occurred trying to write to the file: " + file);
			e.printStackTrace();
			return null;
		}
	}

	// a method for saving all the ANN data (writing it to the files)
	public void save(String saveFolder) throws IOException {
		// write all the values for the settings to the settings.txt file
		PrintStream writer = fileWriter(saveFolder + "settings.txt");
		writer.println(String.valueOf(inputLayerSize));
		writer.println(String.valueOf(hiddenLayerOneSize));
		writer.println(String.valueOf(hiddenLayerTwoSize));
		writer.println(String.valueOf(outputLayerSize));
		writer.println(String.valueOf(min));
		writer.println(String.valueOf(max));
		writer.println(String.valueOf(min));
		writer.println(String.valueOf(max));
		writer.close();

		// write all the values for the biases for the layers
		writer = fileWriter(saveFolder + "biasesLayerOne.txt");
		for (int i = 0; i < hiddenLayerOneSize; i++) {
			writer.println(String.valueOf(biasesLayerOne[i]));
		}
		writer.close();
		writer = fileWriter(saveFolder + "biasesLayerTwo.txt");
		for (int i = 0; i < hiddenLayerTwoSize; i++) {
			writer.println(String.valueOf(biasesLayerTwo[i]));
		}
		writer.close();
		writer = fileWriter(saveFolder + "biasesOutputLayer.txt");
		for (int i = 0; i < outputLayerSize; i++) {
			writer.println(String.valueOf(biasesOutputLayer[i]));
		}
		writer.close();

		// write all the values for the weights for the layers
		writer = fileWriter(saveFolder + "weightsAfterInputLayer.txt");
		for (int i = 0; i < inputLayerSize; i++) {
			for (int j = 0; j < hiddenLayerOneSize; j++) {
				writer.println(String.valueOf(weightsAfterInputLayer[i][j]));
			}
		}
		writer.close();
		writer = fileWriter(saveFolder + "weightsAfterLayerOne.txt");
		for (int i = 0; i < hiddenLayerOneSize; i++) {
			for (int j = 0; j < hiddenLayerTwoSize; j++) {
				writer.println(String.valueOf(weightsAfterLayerOne[i][j]));
			}
		}
		writer.close();
		writer = fileWriter(saveFolder + "weightsAfterLayerTwo.txt");
		for (int i = 0; i < hiddenLayerTwoSize; i++) {
			for (int j = 0; j < outputLayerSize; j++) {
				writer.println(String.valueOf(weightsAfterLayerTwo[i][j]));
			}
		}
		writer.close();
	}

	public ANN(int inputLayerSize, int hiddenLayerOneSize, int hiddenLayerTwoSize, int outputLayerSize, double min,
			double max, boolean readValuesFromSaveFiles, String saveFolder) {
		// if we want to read the ANN from the save files (for the settings)
		if (readValuesFromSaveFiles) {
			// reading the settings
			Scanner fileScanner1 = fileOpener(saveFolder + "settings.txt");
			inputLayerSize = Integer.parseInt(fileScanner1.nextLine());
			hiddenLayerOneSize = Integer.parseInt(fileScanner1.nextLine());
			hiddenLayerTwoSize = Integer.parseInt(fileScanner1.nextLine());
			outputLayerSize = Integer.parseInt(fileScanner1.nextLine());
			min = Double.parseDouble(fileScanner1.nextLine());
			max = Double.parseDouble(fileScanner1.nextLine());
			min = Double.parseDouble(fileScanner1.nextLine());
			max = Double.parseDouble(fileScanner1.nextLine());
			fileScanner1.close();
		}
		// calls all the methods for changing the size of the layers. They will throw
		// exceptions if illegal values are attempted
		changeInputLayerSize(inputLayerSize);
		changeHiddenLayerOneSize(hiddenLayerOneSize);
		changeHiddenLayerTwoSize(hiddenLayerTwoSize);
		changeOutputLayerSize(outputLayerSize);

		// initializing all the biases (they were already initialized as
		// variables above, but now we are assigning them a length and thus a slot in
		// memory)
		biasesLayerOne = new double[hiddenLayerOneSize];
		biasesLayerTwo = new double[hiddenLayerTwoSize];
		biasesOutputLayer = new double[outputLayerSize];

		// initializing all the weights (they were already initialized
		// as variables above, but now we are assigning them a length and thus a slot in
		// memory)
		weightsAfterInputLayer = new double[inputLayerSize][hiddenLayerOneSize];
		weightsAfterLayerOne = new double[hiddenLayerOneSize][hiddenLayerTwoSize];
		weightsAfterLayerTwo = new double[hiddenLayerTwoSize][outputLayerSize];

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
		changeParameters(min, max);

		// if we want to read the ANN from the save files (for the weights and biases)
		if (readValuesFromSaveFiles) {
			// the scanner
			// initialize the scanner and read values for biasesLayerOne[]
			Scanner fileScanner2 = fileOpener(saveFolder + "biasesLayerOne.txt");
			for (int i = 0; i < hiddenLayerOneSize; i++) {
				biasesLayerOne[i] = Double.parseDouble(fileScanner2.nextLine());
			}
			fileScanner2.close();

			// read the values for biasesLayerTwo[]
			fileScanner2 = fileOpener(saveFolder + "biasesLayerTwo.txt");
			for (int i = 0; i < hiddenLayerTwoSize; i++) {
				biasesLayerTwo[i] = Double.parseDouble(fileScanner2.nextLine());
			}
			fileScanner2.close();

			// read the values for biasesOutputLayer[]
			fileScanner2 = fileOpener(saveFolder + "biasesOutputLayer.txt");
			for (int i = 0; i < outputLayerSize; i++) {
				biasesOutputLayer[i] = Double.parseDouble(fileScanner2.nextLine());
			}
			fileScanner2.close();

			// read the values for weightsAfterInputLayer[][]
			fileScanner2 = fileOpener(saveFolder + "weightsAfterInputLayer.txt");
			for (int i = 0; i < inputLayerSize; i++) {
				for (int j = 0; j < hiddenLayerOneSize; j++) {
					weightsAfterInputLayer[i][j] = Double.parseDouble(fileScanner2.nextLine());
				}
			}
			fileScanner2.close();

			// read the values for weightsAfterLayerOne[][]
			fileScanner2 = fileOpener(saveFolder + "weightsAfterLayerOne.txt");
			for (int i = 0; i < hiddenLayerOneSize; i++) {
				for (int j = 0; j < hiddenLayerTwoSize; j++) {
					weightsAfterLayerOne[i][j] = Double.parseDouble(fileScanner2.nextLine());
				}
			}
			fileScanner2.close();

			// read the values for weightsAfterLayerTwo[][]
			fileScanner2 = fileOpener(saveFolder + "weightsAfterLayerTwo.txt");
			for (int i = 0; i < hiddenLayerTwoSize; i++) {
				for (int j = 0; j < outputLayerSize; j++) {
					weightsAfterLayerTwo[i][j] = Double.parseDouble(fileScanner2.nextLine());
				}
			}
			fileScanner2.close();
		} else {
			randomizeOneDimensionalArr(biasesLayerOne);
			randomizeOneDimensionalArr(biasesLayerTwo);
			randomizeOneDimensionalArr(biasesOutputLayer);

			randomizeTwoDimensionalArr(weightsAfterInputLayer);
			randomizeTwoDimensionalArr(weightsAfterLayerOne);
			randomizeTwoDimensionalArr(weightsAfterLayerTwo);
		}
	}

	/*
	 * =============================================================================
	 * ================= Methods Pertaining to Forward Propagation =================
	 * =============================================================================
	 */

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
			arr[i] = (2 * ((arr[i] - min) / (max - min))) - 1;
		}
	}

	// the method that will denormalize the output array
	private void denormalize(double[] arr) {
		// do the denormalization on each element of arr[]
		for (int i = 0; i < arr.length; i++) {
			arr[i] = (((arr[i] + 1) / 2) * (max - min)) + min;
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
			System.out.println("the outputs array after calculation" + arrToString(outputs));
			System.out.println("=====================================================");
		}
	}

	/*
	 * =============================================================================
	 * ================ Methods Pertaining to Backwards Propagation ================
	 * =============================================================================
	 */

	/*
	 * a method to allow the client to change the learning rate (which is set to 0.1
	 * by default)
	 * 
	 * the learning rate must be on the interval [-1.0, 1.0]
	 */
	public void changeLearningRate(double newRate) {
		if (newRate > 1.0 || newRate < -1.0) {
			throw new ArithmeticException("the learning rate must be on the interval [-1.0, 1.0]");
		} else {
			this.learningRate = newRate;
		}
	}

	// a method to un-sigmoid a value (the inverse of the sigmoid function)
	private double sigmoidInverse(double x) {
		// Math.log(x) is ln(x)
		return -1 * Math.log((1 / x) - 1);
	}

	/*
	 * the method that returns the derivative of the sigmoid function
	 * 
	 * check the notebook for the mathematical steps, but:
	 * 
	 * (d/dx)[sigmoid(x)] = sigmoid(x) * (1 - sigmoid(x))
	 */
	private double sigmoidDerivative(double x) {
		return sigmoid(x) * (1 - sigmoid(x));
	}

	// a method to calculate and return the error of a neuron in the output layer
	private double outputNeuronError(double expectedOutput, double nonSigmoidedActualOutput) {
		return (expectedOutput - sigmoid(nonSigmoidedActualOutput)) * sigmoidDerivative(nonSigmoidedActualOutput);
	}

	// a method to calculate and return the error of a neuron in a hidden layer
	private double hiddenNeuronError(double weightFromOutputNeuron, double errorFromOutputNeuron,
			double nonSigmoidedActualOutput) {
		return (weightFromOutputNeuron * errorFromOutputNeuron) * sigmoidDerivative(nonSigmoidedActualOutput);
	}
	
	// a method to copy over weights from the new arrays to the proper weight arrays
	private void copyWeights(double[][] newWeights, double[][] oldWeights) {
		for (int i = 0; i < oldWeights.length; i++) {
			for (int j = 0; j < oldWeights[i].length; j++) {
				oldWeights[i][j] = newWeights[i][j];
			}
		}
	}

	// the method to actually train the ANN
	public void learn(double[] givenInputs, double[] desiredOutputs, int epochs) {
		// repeat the learning process with the training set data for the number of
		// epochs
		for (int i = 0; i < epochs; i++) {
			// an array to hold our actual outputs (which will be used to calculate error)
			double[] actualOutputs = new double[outputLayerSize];

			// normalize the desiredOutputs since we will need them normalized to calculate
			// error
			normalize(desiredOutputs);

			// normalize the inputs
			normalize(givenInputs);
			// this will change all the values in "hiddenLayerOneNodes" to the calculated
			// values
			layerCalculate(givenInputs, hiddenLayerOneNodes, weightsAfterInputLayer, biasesLayerOne);
			// this will change all the values in "hiddenLayerTwoNodes" to the calculated
			// values
			layerCalculate(hiddenLayerOneNodes, hiddenLayerTwoNodes, weightsAfterLayerOne, biasesLayerTwo);
			// this will change all the values in "outputs" to the calculated values
			layerCalculate(hiddenLayerTwoNodes, actualOutputs, weightsAfterLayerTwo, biasesOutputLayer);
			/*
			 * by this part in the code, the actualOutputs[] array should have been changed
			 * to the desired outputs (but still in normalized form)
			 */

			/*
			 * do the actual training (update the weights)
			 */

			// arrays to keep track of the error
			double[] outputLayerError = new double[outputLayerSize];
			double[] hiddenLayerTwoError = new double[hiddenLayerTwoSize];
			double[] hiddenLayerOneError = new double[hiddenLayerOneSize];
			
			// arrays to keep track of the new weights
			double[][] newWeightsAfterLayerTwo = new double[weightsAfterLayerTwo.length][weightsAfterLayerTwo[0].length];
			double[][] newWeightsAfterLayerOne = new double[weightsAfterLayerOne.length][weightsAfterLayerOne[0].length];
			double[][] newWeightsAfterInputLayer = new double[weightsAfterInputLayer.length][weightsAfterInputLayer[0].length];
			
			/*
			 * weightsAfterLayerTwo = new double[hiddenLayerTwoSize][outputLayerSize];
			 * 
			 * adjust the weights that lead to the output layer
			 * 
			 * j keeps track of the hidden layer two node k keeps track of the output layer
			 * node
			 */
			for (int j = 0; j < weightsAfterLayerTwo.length; j++) {
				for (int k = 0; k < weightsAfterLayerTwo[j].length; k++) {
					// store the error in the outputLayerError[] array to be used later
					outputLayerError[k] = outputNeuronError(desiredOutputs[k], sigmoidInverse(actualOutputs[k]));
					// this will sometimes be a positive adjustment and sometimes a negative
					// adjustment
					newWeightsAfterLayerTwo[j][k] += learningRate * hiddenLayerTwoNodes[j] * outputLayerError[k];
				}
			}
			// adjust the weights that lead to the second hidden layer
			for (int j = 0; j < weightsAfterLayerOne.length; j++) {
				for (int k = 0; k < weightsAfterLayerOne[j].length; k++) {
					// store the error to be used later
					/*
					 * THE ERROR: the error is because we are calling the array outputLayerError[]
					 * with index "k". However, index "k" keeps track of the number neuron from the
					 * second hidden layer we are on. Since there are more neurons on the second
					 * hidden layer than the output layer, we end up calling an index of the output
					 * layer that does not exist (and thus we get an ArrayIndexOutOfBoundsException)
					 */
					for (int x = 0; x < outputLayerSize; x++) {
						hiddenLayerTwoError[k] += hiddenNeuronError(weightsAfterLayerTwo[j][x], outputLayerError[x], sigmoidInverse(hiddenLayerTwoNodes[k]));
					}
					// adjust the weight
					newWeightsAfterLayerOne[j][k] += learningRate * hiddenLayerOneNodes[j] * hiddenLayerTwoError[k];
				}
			}
			// adjust the weights that lead to the first hidden layer
			for (int j = 0; j < weightsAfterInputLayer.length; j++) {
				for (int k = 0; k < weightsAfterInputLayer[j].length; k++) {
					// store the error (this time it won't be used later, but maybe in future
					// versions it will and this looks cleaner)
					for (int x = 0; x < hiddenLayerTwoSize; x++ ) {
						hiddenLayerOneError[k] += hiddenNeuronError(weightsAfterLayerOne[j][x], hiddenLayerTwoError[x], sigmoidInverse(hiddenLayerOneNodes[k]));
					}
					// adjust the weight
					newWeightsAfterInputLayer[j][k] += learningRate * givenInputs[j] * hiddenLayerOneError[k];
				}
			}
			
			// copy all the new weights into the arrays for the old weights
			copyWeights(newWeightsAfterLayerTwo, weightsAfterLayerTwo);
			copyWeights(newWeightsAfterLayerOne, weightsAfterLayerOne);
			copyWeights(newWeightsAfterInputLayer, weightsAfterInputLayer);
			
			System.out.println("Number of completed epochs: " + (i + 1));
		}
	}
}