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
	double bias = 1.0;
	// initializing the weights
	double[][] weightsAfterInputLayer, weightsAfterLayerOne, weightsAfterLayerTwo;
	// initializing the nodes
	double[] hiddenLayerOneNodes, hiddenLayerTwoNodes;

	// a boolean to keep track of if we normalized the data or not
	boolean normalized = false;

	// variables to keep track of normalization and denormalization
	double min, max;
	double normalizationRate = 0.9;

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
			throw new ArithmeticException("layers must have at least one node");
		} else {
			inputLayerSize = newInputLayerSize;
		}
	}

	private void changeHiddenLayerOneSize(int newHiddenLayerOneSize) {
		if (newHiddenLayerOneSize <= 0) {
			throw new ArithmeticException("layers must have at least one node");
		} else {
			hiddenLayerOneSize = newHiddenLayerOneSize;
		}
	}

	private void changeHiddenLayerTwoSize(int newHiddenLayerTwoSize) {
		if (newHiddenLayerTwoSize <= 0) {
			throw new ArithmeticException("layers must have at least one node");
		} else {
			hiddenLayerTwoSize = newHiddenLayerTwoSize;
		}
	}

	private void changeOutputLayerSize(int newOutputLayerSize) {
		if (newOutputLayerSize <= 0) {
			throw new ArithmeticException("layers must have at least one node");
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

	// a private method to handle randomly generating doubles on the interval [0.0,
	// 1.0]
	private static double getRandomDouble() {
		Random rand = new Random();
		return rand.nextDouble();
	}

	// to randomize the values in an array between 0 and 1 (used for the biases
	// arrays)
	private static void randomizeOneDimensionalArr(double[] arr) {
		for (int i = 0; i < arr.length; i++) {
			// generates a double on the interval [0.0, 1.0]
			arr[i] = getRandomDouble();

			/*
			 * turns the double into a negative 50% of the time
			 * 
			 * so now the double will be somewhere on the interval [-1.0, 1.0] instead of
			 * the interval [0.0, 1.0]
			 */
			double x = getRandomDouble();
			if (x > 0.5) {
				arr[i] *= -1;
			}
		}
	}

	private static void randomizeTwoDimensionalArr(double[][] arr) {
		for (int i = 0; i < arr.length; i++) {
			for (int j = 0; j < arr[i].length; j++) {
				arr[i][j] = getRandomDouble();
				/*
				 * turns the double into a negative 50% of the time
				 * 
				 * so now the double will be somewhere on the interval [-1.0, 1.0] instead of
				 * the interval [0.0, 1.0]
				 * 
				 * for some reason, evaluating random functions in the if() statement or using
				 * the boolean random functions will always return a value of "true". To get
				 * around this, a random double is generated and stored and then that value is
				 * checked to see if it's greater than 0.5 instead of using a random boolean
				 */
				double x = getRandomDouble();
				if (x > 0.5) {
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
		System.out.println(
				"=========================================================================================================================================================");

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

		// write all the values for the weights for the layers
		writer = fileWriter(saveFolder + "weightsAfterInputLayer.txt");
		for (int i = 0; i < inputLayerSize + 1; i++) {
			for (int j = 0; j < hiddenLayerOneSize; j++) {
				writer.println(String.valueOf(weightsAfterInputLayer[i][j]));
			}
		}
		writer.close();
		writer = fileWriter(saveFolder + "weightsAfterLayerOne.txt");
		for (int i = 0; i < hiddenLayerOneSize + 1; i++) {
			for (int j = 0; j < hiddenLayerTwoSize; j++) {
				writer.println(String.valueOf(weightsAfterLayerOne[i][j]));
			}
		}
		writer.close();
		writer = fileWriter(saveFolder + "weightsAfterLayerTwo.txt");
		for (int i = 0; i < hiddenLayerTwoSize + 1; i++) {
			for (int j = 0; j < outputLayerSize; j++) {
				writer.println(String.valueOf(weightsAfterLayerTwo[i][j]));
			}
		}
		writer.close();

		System.out.println(
				"=========================================================================================================================================================");
	}

	public ANN(int inputLayerSizeFromClient, int hiddenLayerOneSizeFromClient, int hiddenLayerTwoSizeFromClient,
			int outputLayerSizeFromClient, double minFromClient, double maxFromClient, boolean readValuesFromSaveFiles,
			String saveFolder) {
		// if we want to read the ANN from the save files (for the settings)
		if (readValuesFromSaveFiles) {
			// reading the settings
			Scanner fileScanner1 = fileOpener(saveFolder + "settings.txt");
			this.inputLayerSize = Integer.parseInt(fileScanner1.nextLine());
			this.hiddenLayerOneSize = Integer.parseInt(fileScanner1.nextLine());
			this.hiddenLayerTwoSize = Integer.parseInt(fileScanner1.nextLine());
			this.outputLayerSize = Integer.parseInt(fileScanner1.nextLine());
			this.min = Double.parseDouble(fileScanner1.nextLine());
			this.max = Double.parseDouble(fileScanner1.nextLine());
			fileScanner1.close();
		} else {
			// changes the min and max for normalization and denormalization
			changeParameters(minFromClient, maxFromClient);
		}
		// calls all the methods for changing the size of the layers. They will throw
		// exceptions if illegal values are attempted
		changeInputLayerSize(inputLayerSizeFromClient);
		changeHiddenLayerOneSize(hiddenLayerOneSizeFromClient);
		changeHiddenLayerTwoSize(hiddenLayerTwoSizeFromClient);
		changeOutputLayerSize(outputLayerSizeFromClient);

		/*
		 * initializing all the weights (they were already initialized as variables
		 * above, but now we are assigning them a length and thus a slot in memory)
		 * 
		 * "+ 1" is added to each first index of the weight arrays to make room for a
		 * node to represent the biases (but since the biases will not get input as part
		 * of their calculation, the second index of the weight arrays needn't contain a
		 * slot for the biases)
		 */
		this.weightsAfterInputLayer = new double[this.inputLayerSize + 1][this.hiddenLayerOneSize];
		this.weightsAfterLayerOne = new double[this.hiddenLayerOneSize + 1][this.hiddenLayerTwoSize];
		this.weightsAfterLayerTwo = new double[this.hiddenLayerTwoSize + 1][this.outputLayerSize];

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
		this.hiddenLayerOneNodes = new double[this.hiddenLayerOneSize];
		this.hiddenLayerTwoNodes = new double[this.hiddenLayerTwoSize];

		// if we want to read the ANN from the save files (for the weights and biases)
		if (readValuesFromSaveFiles) {
			// the scanner
			// read the values for weightsAfterInputLayer[][]
			Scanner fileScanner2 = fileOpener(saveFolder + "weightsAfterInputLayer.txt");
			for (int i = 0; i < this.inputLayerSize + 1; i++) {
				for (int j = 0; j < this.hiddenLayerOneSize; j++) {
					this.weightsAfterInputLayer[i][j] = Double.parseDouble(fileScanner2.nextLine());
				}
			}
			fileScanner2.close();

			// read the values for weightsAfterLayerOne[][]
			fileScanner2 = fileOpener(saveFolder + "weightsAfterLayerOne.txt");
			for (int i = 0; i < this.hiddenLayerOneSize + 1; i++) {
				for (int j = 0; j < this.hiddenLayerTwoSize; j++) {
					this.weightsAfterLayerOne[i][j] = Double.parseDouble(fileScanner2.nextLine());
				}
			}
			fileScanner2.close();

			// read the values for weightsAfterLayerTwo[][]
			fileScanner2 = fileOpener(saveFolder + "weightsAfterLayerTwo.txt");
			for (int i = 0; i < this.hiddenLayerTwoSize + 1; i++) {
				for (int j = 0; j < this.outputLayerSize; j++) {
					this.weightsAfterLayerTwo[i][j] = Double.parseDouble(fileScanner2.nextLine());
				}
			}
			fileScanner2.close();
		} else {
			randomizeTwoDimensionalArr(this.weightsAfterInputLayer);
			randomizeTwoDimensionalArr(this.weightsAfterLayerOne);
			randomizeTwoDimensionalArr(this.weightsAfterLayerTwo);
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
		return (1.0 / (1.0 + (1.0 / Math.exp(x))));
	}

	// the method that will calculate a single layer of node values
	private void layerCalculate(double[] inputs, double[] targets, double[][] weights) {
		for (int i = 0; i < targets.length; i++) {
			// this will be the sum of all the inputs times all the weights plus a bias
			double sum = 0;
			for (int j = 0; j < inputs.length; j++) {
				sum += (inputs[j] * weights[j][i]);
			}
			// add the bias to "sum"
			sum += this.bias * weights[inputs.length][i];
			// reasign the "sum" variable as the sigmoid of "sum" if it isn't the output
			// layer
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
			arr[i] = ((((arr[i] - this.max) - this.min) / (this.max - this.min)) * this.normalizationRate) + 0.5;
		}
	}

	// the method that will denormalize the output array
	private void denormalize(double[] arr) {
		// do the denormalization on each element of arr[]
		for (int i = 0; i < arr.length; i++) {
			arr[i] = (((arr[i] - 0.5) * (this.max - this.min)) / this.normalizationRate) + this.min + this.max;
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
			layerCalculate(inputs, hiddenLayerOneNodes, weightsAfterInputLayer);
			// this will change all the values in "hiddenLayerTwoNodes" to the calculated
			// values
			layerCalculate(hiddenLayerOneNodes, hiddenLayerTwoNodes, weightsAfterLayerOne);
			// this will change all the values in "outputs" to the calculated values
			layerCalculate(hiddenLayerTwoNodes, outputs, weightsAfterLayerTwo);
			// by this part in the code, the outputs[] array should have been changed to the
			// desired outputs (but still in normalized form)

			// denormalize the outputs
//			denormalize(outputs);

//			System.out.println("=====================================================");
//			System.out.println("the outputs array after calculation" + arrToString(outputs));
//			System.out.println("=====================================================");
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
		return (-1.0) * Math.log((1.0 / x) - 1.0);
	}

	/*
	 * the method that returns the derivative of the sigmoid function
	 * 
	 * check the notebook for the mathematical steps, but:
	 * 
	 * (d/dx)[sigmoid(x)] = sigmoid(x) * (1 - sigmoid(x))
	 */
	private double sigmoidDerivative(double x) {
		return sigmoid(x) * (1.0 - sigmoid(x));
	}

	// a method to calculate and return the error of a neuron in the output layer
	private double outputNeuronError(double expectedOutput, double nonSigmoidedActualOutput, double inputNode) {
//		System.out.println("===================================================");
//		System.out.println("expectedOutput = " + expectedOutput);
//		System.out.println("nonSigmoidedActualOutput = " + nonSigmoidedActualOutput);
//		System.out.println("sigmoid = " + sigmoid(nonSigmoidedActualOutput));
//		System.out.println("sigmoidDerivative = " + sigmoidDerivative(nonSigmoidedActualOutput));
//		System.out.println("===================================================");
		return (expectedOutput - sigmoid(nonSigmoidedActualOutput)) * inputNode;
		// * sigmoidDerivative(nonSigmoidedActualOutput)
	}

	// a method to calculate and return the error of a neuron in a hidden layer
	private double hiddenNeuronError(double weightFromOutputNeuron, double nonSigmoidedActualOutput, double inputNode,
			double errorFromOutputNeuron) {
		return weightFromOutputNeuron * inputNode * errorFromOutputNeuron;
		// * sigmoidDerivative(nonSigmoidedActualOutput)
	}

	// a method to copy over weights from the new arrays to the proper weight arrays
	private void copyWeights(double[][] newWeights, double[][] oldWeights) {
		for (int i = 0; i < oldWeights.length; i++) {
			for (int j = 0; j < oldWeights[i].length; j++) {
				oldWeights[i][j] = newWeights[i][j];
			}
		}
	}
	
	/*
	 * a method to shuffle the inputs for learning
	 * 
	 * this method takes in a 2D array since the inputs for learning are put in a 2D array
	 */
	private void shuffleInputsAndOutputs(double[][] inputs, double[][] outputs) {
		Random rgen = new Random();
		
		for (int i = 0; i < inputs.length; i++) {
			// generate the random position
			int randomPosition = rgen.nextInt(inputs.length);
			
			// switch to elements in the inputs[][] array
			double[] temp = inputs[i];
			inputs[i] = inputs[randomPosition];
			inputs[randomPosition] = temp;
			
			// switch two elements in the outputs[][] array (with the same indices of the switched elements in the inputs[][] array)
			temp = outputs[i];
			outputs[i] = outputs[randomPosition];
			outputs[randomPosition] = temp;
		}
	}

	// the method to actually train the ANN
	public void learn(double[][] givenInputs, double[][] desiredOutputs, int dataSetSize) {
		// randomize the testing data before passing it into the ANN
		shuffleInputsAndOutputs(givenInputs, desiredOutputs);
		
		// repeat the learning process with the training set data for the number of
		// data points in the data set
		for (int i = 0; i < dataSetSize; i++) {
			// an array to hold our actual outputs (which will be used to calculate error)
			double[] actualOutputs = new double[this.outputLayerSize];

			// normalize the desiredOutputs since we will need them normalized to calculate
			// error
			normalize(desiredOutputs[i]);

			// normalize the inputs
			normalize(givenInputs[i]);

			// this will change all the values in "hiddenLayerOneNodes" to the calculated
			// values
			layerCalculate(givenInputs[i], this.hiddenLayerOneNodes, this.weightsAfterInputLayer);
			// this will change all the values in "hiddenLayerTwoNodes" to the calculated
			// values
			layerCalculate(this.hiddenLayerOneNodes, this.hiddenLayerTwoNodes, this.weightsAfterLayerOne);
			// this will change all the values in "outputs" to the calculated values
			layerCalculate(this.hiddenLayerTwoNodes, actualOutputs, this.weightsAfterLayerTwo);
			/*
			 * by this part in the code, the actualOutputs[] array should have been changed
			 * to the desired outputs (but still in normalized form)
			 */

			/*
			 * do the actual training (update the weights)
			 */

			// arrays to keep track of the error
			double[] outputLayerError = new double[this.outputLayerSize];
			double[] hiddenLayerTwoError = new double[this.hiddenLayerTwoSize];
			double[] hiddenLayerOneError = new double[this.hiddenLayerOneSize];

			// arrays to keep track of the new weights
			double[][] newWeightsAfterLayerTwo = new double[this.weightsAfterLayerTwo.length][this.weightsAfterLayerTwo[0].length];
			double[][] newWeightsAfterLayerOne = new double[this.weightsAfterLayerOne.length][this.weightsAfterLayerOne[0].length];
			double[][] newWeightsAfterInputLayer = new double[this.weightsAfterInputLayer.length][this.weightsAfterInputLayer[0].length];
			/*
			 * adjust the weights that lead to the output layer
			 * 
			 * j keeps track of the hidden layer two node k keeps track of the output layer
			 * node
			 * 
			 * we subtract 1 from the length of the weights after layer two so that we don't
			 * try to mess with the weight that connects to the bias (which doesn't have a
			 * corresponding hidden node)
			 */
			for (int j = 0; j < this.weightsAfterLayerTwo.length; j++) {
				for (int k = 0; k < this.weightsAfterLayerTwo[j].length; k++) {
					// if we are on the weight that connects to the bias, pass in the bias as the
					// value for the node from the previous layer. Otherwise, just pass in the
					// actual value from the previous layer
					double previousLayerNode;
					if (j == this.weightsAfterLayerTwo.length - 1) {
						previousLayerNode = this.bias;
					} else {
						previousLayerNode = this.hiddenLayerTwoNodes[j];
					}

					// store the error in the outputLayerError[] array to be used later
					outputLayerError[k] = outputNeuronError(desiredOutputs[i][k], sigmoidInverse(actualOutputs[k]),
							previousLayerNode);

//					System.out.println("outputLayerError[" + k + "] = " + outputLayerError[k]);

					// the following will sometimes be a positive adjustment and sometimes a
					// negative adjustment
					newWeightsAfterLayerTwo[j][k] = weightsAfterLayerTwo[j][k]
							+ (Math.abs(weightsAfterLayerTwo[j][k]) * this.learningRate * outputLayerError[k]);

					// make sure that the new weight stays on the interval [-1.0, 1.0]
					if (newWeightsAfterLayerTwo[j][k] > 1.0) {
						newWeightsAfterLayerTwo[j][k] = 1.0;
					} else if (newWeightsAfterLayerTwo[j][k] < -1.0) {
						newWeightsAfterLayerTwo[j][k] = -1.0;
					}
				}
			}
			// adjust the weights that lead to the second hidden layer
			for (int j = 0; j < this.weightsAfterLayerOne.length; j++) {
				for (int k = 0; k < this.weightsAfterLayerOne[j].length; k++) {
					// if we are on the weight that connects to the bias, pass in the bias as the
					// value for the node from the previous layer. Otherwise, just pass in the
					// actual value from the previous layer
					double previousLayerNode;
					if (j == this.weightsAfterLayerOne.length - 1) {
						previousLayerNode = this.bias;
					} else {
						previousLayerNode = this.hiddenLayerOneNodes[j];
					}

					// store the error to be used later
					for (int x = 0; x < this.outputLayerSize; x++) {
						hiddenLayerTwoError[k] += hiddenNeuronError(this.weightsAfterLayerTwo[k][x],
								sigmoidInverse(this.hiddenLayerTwoNodes[k]), previousLayerNode, outputLayerError[x]);
					}

					// adjust the weight
					newWeightsAfterLayerOne[j][k] = weightsAfterLayerOne[j][k]
							+ (Math.abs(weightsAfterLayerOne[j][k]) * this.learningRate * hiddenLayerTwoError[k]);

					// make sure that the new weight stays on the interval [-1.0, 1.0]
					if (newWeightsAfterLayerOne[j][k] > 1.0) {
						newWeightsAfterLayerOne[j][k] = 1.0;
					} else if (newWeightsAfterLayerOne[j][k] < -1.0) {
						newWeightsAfterLayerOne[j][k] = -1.0;
					}
				}
			}
			// adjust the weights that lead to the first hidden layer
			for (int j = 0; j < this.weightsAfterInputLayer.length; j++) {
				for (int k = 0; k < this.weightsAfterInputLayer[j].length; k++) {
					// if we are on the weight that connects to the bias, pass in the bias as the
					// value for the node from the previous layer. Otherwise, just pass in the
					// actual value from the previous layer
					double previousLayerNode;
					if (j == this.weightsAfterInputLayer.length - 1) {
						previousLayerNode = this.bias;
					} else {
						previousLayerNode = givenInputs[i][j];
					}

					// store the error (this time it won't be used later, but maybe in future
					// versions it will and this looks cleaner)
					for (int x = 0; x < this.hiddenLayerTwoSize; x++) {
						// double weightFromOutputNeuron, double nonSigmoidedActualOutput, double
						// inputNode, double errorFromOutputNeuron
						hiddenLayerOneError[k] += hiddenNeuronError(this.weightsAfterLayerOne[k][x],
								sigmoidInverse(this.hiddenLayerOneNodes[k]), previousLayerNode, hiddenLayerTwoError[x]);
					}

					// adjust the weight
					newWeightsAfterInputLayer[j][k] = weightsAfterInputLayer[j][k]
							+ (Math.abs(weightsAfterInputLayer[j][k]) * this.learningRate * hiddenLayerOneError[k]);

					// make sure that the new weight stays on the interval [-1.0, 1.0]
					if (newWeightsAfterInputLayer[j][k] > 1.0) {
						newWeightsAfterInputLayer[j][k] = 1.0;
					} else if (newWeightsAfterInputLayer[j][k] < -1.0) {
						newWeightsAfterInputLayer[j][k] = -1.0;
					}
				}
			}

			// copy all the new weights into the arrays for the old weights
			copyWeights(newWeightsAfterLayerTwo, this.weightsAfterLayerTwo);
			copyWeights(newWeightsAfterLayerOne, this.weightsAfterLayerOne);
			copyWeights(newWeightsAfterInputLayer, this.weightsAfterInputLayer);
		}
	}

	/*
	 * =============================================================================
	 * =========================== Miscellaneous Methods ===========================
	 * =============================================================================
	 */

	public void test(double[] arr) {
		normalize(arr);
		denormalize(arr);
	}
}