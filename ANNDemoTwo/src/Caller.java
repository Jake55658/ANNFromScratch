import java.io.IOException;

public class Caller {
	// the main() method
	public static void main(String args[]) throws IOException {
		// a String variable to hold the folder location for the ANN save files
		String firstANNSaveDataLocation = "D:\\\\\\\\Coding Shtuyot\\\\\\\\Git Data From Eclipse\\\\\\\\HDRepository\\\\\\\\ANNDemoTwo\\\\\\\\src\\\\\\\\ANNData\\\\\\\\firstANN\\\\\\\\";
		
		// an instance of the ANN class (from the ANN.java file)
		ANN firstANN = new ANN(1, 20, 20, 1, -10000, 10000, true, firstANNSaveDataLocation);
		
		/*
		 * the input array for the NN
		 * 
		 * we are defining it and assigning it values here since we are assigning it
		 * values here, the size of the array is automatically assigned according to how
		 * many values are passed in
		 */
		double[] firstInputs = new double[] {500};

		// the output array for the NN
		// we must assign the outputs[] array a size for the NN to work
		double[] firstOutputs = new double[1];

		// using our instance of an ANN to calculate outputs based on the inputs
		firstANN.calculate(firstInputs, firstOutputs);
		//firstANN.save(firstANNSaveDataLocation);
		
		System.out.println("Finished: 1");
		
		/*
		 * the learning:
		 */
		// generate the test data
		int epochs = 100;
		double[] givenInputs = new double[1];
		double[] desiredOutputs = new double[1];
		// pass the testing data in
		for (int i = 0; i < 1000; i++) {
			givenInputs[0] = (i * 2) - 1000;
			desiredOutputs[0] = (i * 2) - 1000;
		}
		// train the ANN
		firstANN.learn(givenInputs, desiredOutputs, epochs);
		
		/*
		 * verifying that the training worked and saving the trained ANN
		 */
		double[] secondInputs = new double[] {500};
		double[] secondOutputs = new double[1];
		firstANN.calculate(secondInputs, secondOutputs);
		
		System.out.println("firstOutputs = " + firstOutputs[0]);
		System.out.println("secondOutputs = " + secondOutputs[0]);
	}
}