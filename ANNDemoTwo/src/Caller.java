import java.io.IOException;

public class Caller {
	// the main() method
	public static void main(String args[]) throws IOException {
		// a String variable to hold the folder location for the ANN save files
		String firstANNSaveDataLocation = "D:\\\\\\\\Coding Shtuyot\\\\\\\\Git Data From Eclipse\\\\\\\\HDRepository\\\\\\\\ANNDemoTwo\\\\\\\\src\\\\\\\\ANNData\\\\\\\\firstANN\\\\\\\\";

		int min = -10000;
		int max = 10000;

		// an instance of the ANN class (from the ANN.java file)
		ANN firstANN = new ANN(1, 20, 20, 1, min, max, true, firstANNSaveDataLocation);

		/*
		 * the input array for the NN
		 * 
		 * we are defining it and assigning it values here since we are assigning it
		 * values here, the size of the array is automatically assigned according to how
		 * many values are passed in
		 */
		double[] firstInputs = new double[] { 5000 };

		// the output array for the NN
		// we must assign the outputs[] array a size for the NN to work
		double[] firstOutputs = new double[1];

		// using our instance of an ANN to calculate outputs based on the inputs
		firstANN.calculate(firstInputs, firstOutputs);
		firstANN.save(firstANNSaveDataLocation);

		/*
		 * the learning:
		 */
		boolean learn = false;
		if (learn) {
			// generate the test data
			int dataSetSize = 1000;
			// the first index is each data point, the second index is the data in each data
			// point
			double[][] givenInputs = new double[dataSetSize][1];
			double[][] desiredOutputs = new double[dataSetSize][1];
			// pass the testing data in

			// since we are using the "i" in the for() loop to keep track of the value to
			// pass into the inputs and outputs, a separate counter variable is used to keep
			// track of the index of the input and output arrays
			double value = min + ((max - min) / (dataSetSize + 1.0));
			for (int i = 0; i < dataSetSize; i++) {
				givenInputs[i][0] = value;
				desiredOutputs[i][0] = value;
				value += (max - min) / (dataSetSize + 1.0);
			}
			// train the ANN
			int epochs = 100;
			for (int i = 0; i < epochs; i++) {
				firstANN.learn(givenInputs, desiredOutputs, dataSetSize);
			}

			/*
			 * verifying that the training worked and saving the trained ANN
			 */
			double[] secondInputs = new double[] { 5000 };
			double[] secondOutputs = new double[1];
			firstANN.save(firstANNSaveDataLocation);
			firstANN.calculate(secondInputs, secondOutputs);
			System.out.println("firstInputs = " + firstInputs[0]);
			System.out.println("firstOutputs = " + firstOutputs[0]);
			System.out.println("secondInputs = " + secondInputs[0]);
			System.out.println("secondOutputs = " + secondOutputs[0]);
		}
	}
}