package bug.developer.recommend.methods;

import java.util.Arrays;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.sun.corba.se.spi.orbutil.fsm.Action;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.Utils;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.lazy.MultiLabelKNN;
import mulan.core.MulanRuntimeException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.measure.FMeasure;

public class DevRec extends MultiLabelKNN {
	protected double smooth;
	private double[] PriorProbabilities;
	private double[] PriorNProbabilities;
	private double[][] CondProbabilities;
	private double[][] CondNProbabilities;
	private int sampleNum;

	public double[] getPriorProbabilities() {
		return PriorProbabilities;
	}

	public double[] getPriorNProbabilities() {
		return PriorNProbabilities;
	}

	public double[][] getCondProbabilities() {
		return CondProbabilities;
	}

	public double[][] getCondNProbabilities() {
		return CondNProbabilities;
	}

	private int topk;
	private int metric;
	private int numInstance;
	private double[][] productScores;
	private double[][] compScores;
	private double alpha = 0;
	private double beta1 = 0;
	private double beta2 = 0;
	private double beta3 = 0;
	private double beta4 = 0;
	private int[] fixBugArr;

	public double[][] getProductScores() {
		return productScores;
	}

	public void setProductScores(double[][] productScores) {
		this.productScores = productScores;
	}

	public double[][] getCompScores() {
		return compScores;
	}

	public void setCompScores(double[][] compScores) {
		this.compScores = compScores;
	}

	public double getAlpha() {
		return alpha;
	}

	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}

	public double getBeta1() {
		return beta1;
	}

	public void setBeta1(double beta1) {
		this.beta1 = beta1;
	}

	public double getBeta2() {
		return beta2;
	}

	public void setBeta2(double beta2) {
		this.beta2 = beta2;
	}

	public double getBeta3() {
		return beta3;
	}

	public void setBeta3(double beta3) {
		this.beta3 = beta3;
	}

	public double getBeta4() {
		return beta4;
	}

	public void setBeta4(double beta4) {
		this.beta4 = beta4;
	}

	public double[][] getTermFixerScores() {
		return termFixerScores;
	}

	public void setTermFixerScores(double[][] termFixerScores) {
		this.termFixerScores = termFixerScores;
	}

	public double[][] getTopicDiscusserScore() {
		return topicDiscusserScore;
	}

	public void setTopicDiscusserScore(double[][] topicDiscusserScore) {
		this.topicDiscusserScore = topicDiscusserScore;
	}

	private int[] termBugArr;
	private int[][] termFixerBugArr;
	private double[][] termFixerScores;

	private double[][] topicDiscusserScore;
	private double[][] topicDiscusser;
	private double[] topicBugReports;
	private Instances trainDatasets;
	private int[] topicIndex;
	private int[] termIndex;
	private double[][] productCount;
	private double[][] compCount;
	private int topicNum;
	private int termSelections;
	private int[][] topTermsIndex;

	@Override
	public TechnicalInformation getTechnicalInformation() {
 
		return null;
	}

	public DevRec(int numOfNeighbors, int topk, int metric,
			int topicNum, int termSelections) {
		super(numOfNeighbors);
		this.topk = topk;
		this.metric = metric;
		this.topicNum = topicNum;
		this.smooth = 1.0;
		this.termSelections = termSelections;
	}

	@Override
	protected void buildInternal(MultiLabelInstances trainingSet)
			throws Exception {
		super.buildInternal(trainingSet);

		trainDatasets = new Instances(trainingSet.getDataSet());
		numInstance = trainDatasets.numInstances();
		topicIndex = new int[topicNum];
		termIndex = new int[featureIndices.length - topicNum - 2];

		for (int i = 0; i < topicNum; i++) {
			topicIndex[i] = featureIndices[i];
		}

		for (int i = 0; i < featureIndices.length - topicNum - 2; i++) {
			termIndex[i] = featureIndices[i + topicNum];
		}

		termBugArr = new int[featureIndices.length - topicNum - 2];
		fixBugArr = new int[numLabels];
		termFixerBugArr = new int[featureIndices.length - topicNum - 2][numLabels];
		termFixerScores = new double[featureIndices.length - topicNum - 2][numLabels];

		topicBugReports = new double[topicNum];
		topicDiscusser = new double[topicNum][numLabels];
		topicDiscusserScore = new double[topicNum][numLabels];
		topTermsIndex = new int[numLabels][termSelections];

		numInstance = trainDatasets.numInstances();

		for (int curr = 0; curr < numInstance; curr++) {
			Instance currIns = trainDatasets.instance(curr);

			for (int i = 0; i < topicNum; i++) {
				double topicValue = Double.parseDouble(currIns
						.toString(topicIndex[i]));
				if (topicValue > 0) {
					topicBugReports[i] += topicValue;
					for (int j = 0; j < numLabels; j++) {
						double discuss = Double.parseDouble(currIns
								.toString(labelIndices[j]));
						if (discuss > 0) {

							 
							topicDiscusser[i][j] += topicValue;
						}
					}
				}

			}

			for (int i = 0; i < featureIndices.length - topicNum - 2; i++) {

				double termValue = Double.parseDouble(currIns
						.toString(termIndex[i]));
				if (termValue > 0) {
					termBugArr[i]++;
					for (int j = 0; j < numLabels; j++) {
						double fixerValue = Double.parseDouble(currIns
								.toString(labelIndices[j]));
						if (fixerValue > 0) {

							termFixerBugArr[i][j]++;
							fixBugArr[j]++;
						}
					}
				}

			}

		}

		for (int i = 0; i < topicNum; i++)
			for (int j = 0; j < numLabels; j++) {

				double re = (double) (topicBugReports[i]);
				if (re != 0) {
					topicDiscusserScore[i][j] = ((double) (topicDiscusser[i][j]) / re);

				} else {
					topicDiscusserScore[i][j] = 0;
				}
			}

		for (int i = 0; i < featureIndices.length - topicNum - 2; i++)
			for (int j = 0; j < numLabels; j++) {

				double re = (double) (termBugArr[i] + fixBugArr[j] - termFixerBugArr[i][j]);
				if (re != 0) {
					termFixerScores[i][j] = ((double) (termFixerBugArr[i][j]) / re);

				} else {
					termFixerScores[i][j] = 0;
				}
			}

		for (int j = 0; j < numLabels; j++) {
			double[] tmp = new double[featureIndices.length - topicNum - 2];
			for (int p = 0; p < featureIndices.length - topicNum - 2; p++) {
				tmp[p] = termFixerScores[p][j];
			}

			topTermsIndex[j] = findTermIndex(tmp, termSelections);
		}

		int numComponent = trainDatasets.attribute(featureIndices.length - 1)
				.numValues();
		int numProd = trainDatasets.attribute(featureIndices.length - 2)
				.numValues();
	 

		productScores = new double[numProd][numLabels];
		compScores = new double[numComponent][numLabels];

		productCount = new double[numProd][numLabels];
		compCount = new double[numComponent][numLabels];

		for (int i = 0; i < numInstance; i++) {
			Instance currIns = trainDatasets.instance(i);
			int compValue = (int) currIns
					.value(featureIndices[featureIndices.length - 1]);
			int prodValue = (int) currIns
					.value(featureIndices[featureIndices.length - 2]);

			for (int j = 0; j < numLabels; j++) {
				double fixerValue = Double.parseDouble(currIns
						.toString(labelIndices[j]));
				if (fixerValue > 0) {
					compCount[compValue][j]++;
					productCount[prodValue][j]++;

				}
			}
		}
		double[] sumproduct = new double[numProd];
		double[] sumComp = new double[numComponent];
		for (int i = 0; i < numProd; i++) {
			for (int j = 0; j < numLabels; j++) {
				sumproduct[i] += productCount[i][j];
				sumComp[i] += compCount[i][j];
			}
		}

		for (int i = 0; i < numProd; i++) {
			for (int j = 0; j < numLabels; j++) {
				if (sumproduct[i] != 0) {
					productScores[i][j] = productCount[i][j] / sumproduct[i];
				} else {
					productScores[i][j] = 0;
				}
				if (sumComp[i] != 0) {
					compScores[i][j] = compCount[i][j] / sumComp[i];
				} else {
					compScores[i][j] = 0;
				}

			}
		}
		PriorProbabilities = new double[numLabels];
		PriorNProbabilities = new double[numLabels];
		CondProbabilities = new double[numLabels][numOfNeighbors + 1];
		CondNProbabilities = new double[numLabels][numOfNeighbors + 1];

		ComputePrior();
		ComputeCond();

		sampleNum = numInstance / 10;
		int[] sampleNumber = new int[sampleNum];

		int intRd = 0;
		int count = 0;
		int flag = 0;
		while (count < sampleNum) {
			Random rdm = new Random(System.currentTimeMillis());
			intRd = Math.abs(rdm.nextInt()) % numInstance;
			for (int i = 0; i < count; i++) {
				if (sampleNumber[i] == intRd) {
					flag = 1;
					break;
				} else {
					flag = 0;
				}
			}
			if (flag == 0) {
				sampleNumber[count] = intRd;
				count++;
			}
		}

		int[][] instanceLabels = new int[sampleNum][numLabels];

		double[][] trainTopicScores = new double[sampleNum][numLabels];
		double[][] trainBugzieScores = new double[sampleNum][numLabels];
		double[][] trainProdScores = new double[sampleNum][numLabels];
		double[][] trainComponentScores = new double[sampleNum][numLabels];
		double[][] trainMlknnScores = new double[sampleNum][numLabels];
		for (int curr = 0; curr < sampleNum; curr++) {
			Instance currIns = trainDatasets.instance(sampleNumber[curr]);
			trainTopicScores[curr] = getTopicScores(currIns);
			trainBugzieScores[curr] = getBugzieScores(currIns);

			for (int j = 0; j < numLabels; j++) {
				double discuss = Double.parseDouble(currIns
						.toString(labelIndices[j]));
				if (discuss > 0) {

					instanceLabels[curr][j] = 1;

				}
			}

		}

		for (int curr = 0; curr < sampleNum; curr++) {
			Instance currIns = trainDatasets.instance(sampleNumber[curr]);

			trainMlknnScores[curr] = getMlkNNScores(currIns);
			int compValue = (int) currIns.value(featureIndices.length - 1);

			int prodValue = (int) currIns.value(featureIndices.length - 2);
			for (int j = 0; j < numLabels; j++) {

				trainProdScores[curr][j] = productScores[prodValue][j];
				trainComponentScores[curr][j] = compScores[compValue][j];

				double discuss = Double.parseDouble(currIns
						.toString(labelIndices[j]));

			}
		}

		double sumAccuracy = 0;

		int iteration = 10;
	 

		for (int currIter = 0; currIter < iteration; currIter++) {
			double[] tmp = new double[5];
			tmp[0] = Math.random();
			tmp[1] = Math.random();
			tmp[2] = Math.random();
			tmp[3] = Math.random();
			tmp[4] = Math.random();
			for (int p = 0; p < 5; p++) {

				for (int i = 0; i < 101; i++) {
					tmp[p] = i * 0.01;
					double[][] tempScores = new double[sampleNum][numLabels];
					int[][] tempdis = new int[sampleNum][numLabels];
					double[][] tempfinalScores = new double[sampleNum][numLabels];
					for (int j = 0; j < sampleNum; j++)
						for (int k = 0; k < numLabels; k++) {
							tempScores[j][k] = tmp[0] * trainTopicScores[j][k]
									+ tmp[1] * trainBugzieScores[j][k] + tmp[2]
									* trainComponentScores[j][k] + tmp[3]
									* trainProdScores[j][k] + tmp[4]
									* trainMlknnScores[j][k];
						}

					for (int j = 0; j < sampleNum; j++) {
						tempfinalScores[j] = findtopkMatch(tempScores[j],
								metric);
					}
					for (int j = 0; j < sampleNum; j++)
						for (int k = 0; k < numLabels; k++) {
							if (tempfinalScores[j][k] > 0) {
								tempdis[j][k] = 1;
							} else {
								tempdis[j][k] = 0;
							}
						}

					double tempAcc = getTopkHit(instanceLabels, tempdis);

					if (tempAcc > sumAccuracy) {
						sumAccuracy = tempAcc;

						beta1 = tmp[0];
						beta2 = tmp[1];
						beta3 = tmp[2];
						beta4 = tmp[3];
						alpha = tmp[4];
					}
				}
			}

		}

		 

	}

	private void ComputePrior() {
		for (int i = 0; i < numLabels; i++) {
			int temp_Ci = 0;
			for (int j = 0; j < train.numInstances(); j++) {
				double value = Double.parseDouble(train.attribute(
						labelIndices[i]).value(
						(int) train.instance(j).value(labelIndices[i])));
				if (Utils.eq(value, 1.0)) {
					temp_Ci++;
				}
			}
			PriorProbabilities[i] = (smooth + temp_Ci)
					/ (smooth * 2 + train.numInstances());
			PriorNProbabilities[i] = 1 - PriorProbabilities[i];
		}
	}

	 
	private void ComputeCond() throws Exception {
		int[][] temp_Ci = new int[numLabels][numOfNeighbors + 1];
		int[][] temp_NCi = new int[numLabels][numOfNeighbors + 1];

		for (int i = 0; i < train.numInstances(); i++) {

			Instances knn = new Instances(lnn.kNearestNeighbours(
					train.instance(i), numOfNeighbors));

			// now compute values of temp_Ci and temp_NCi for every class label
			for (int j = 0; j < numLabels; j++) {

				int aces = 0; // num of aces in Knn for j
				for (int k = 0; k < numOfNeighbors; k++) {
					double value = Double.parseDouble(train.attribute(
							labelIndices[j]).value(
							(int) knn.instance(k).value(labelIndices[j])));
					if (Utils.eq(value, 1.0)) {
						aces++;
					}
				}
				// raise the counter of temp_Ci[j][aces] and temp_NCi[j][aces]
				// by 1
				if (Utils.eq(
						Double.parseDouble(train.attribute(labelIndices[j])
								.value((int) train.instance(i).value(
										labelIndices[j]))), 1.0)) {
					temp_Ci[j][aces]++;
				} else {
					temp_NCi[j][aces]++;
				}
			}
		}

		// compute CondProbabilities[i][..] for labels based on temp_Ci[]
		for (int i = 0; i < numLabels; i++) {
			int temp1 = 0;
			int temp2 = 0;
			for (int j = 0; j < numOfNeighbors + 1; j++) {
				temp1 += temp_Ci[i][j];
				temp2 += temp_NCi[i][j];
			}
			for (int j = 0; j < numOfNeighbors + 1; j++) {
				CondProbabilities[i][j] = (smooth + temp_Ci[i][j])
						/ (smooth * (numOfNeighbors + 1) + temp1);
				CondNProbabilities[i][j] = (smooth + temp_NCi[i][j])
						/ (smooth * (numOfNeighbors + 1) + temp2);
			}
		}
	}

	private double[] getMlkNNScores(Instance instance) {
		double[] confidences = new double[numLabels];
		boolean[] predictions = new boolean[numLabels];

		Instances knn = null;
		try {
			knn = new Instances(
					lnn.kNearestNeighbours(instance, numOfNeighbors));
		} catch (Exception ex) {
			Logger.getLogger(DevRec.class.getName()).log(Level.SEVERE,
					null, ex);
		}

		for (int i = 0; i < numLabels; i++) {
			// compute sum of aces in KNN
			int aces = 0; // num of aces in Knn for i
			for (int k = 0; k < numOfNeighbors; k++) {
				double value = Double.parseDouble(train.attribute(
						labelIndices[i]).value(
						(int) knn.instance(k).value(labelIndices[i])));
				if (Utils.eq(value, 1.0)) {
					aces++;
				}
			}
			double Prob_in = PriorProbabilities[i] * CondProbabilities[i][aces];
			double Prob_out = PriorNProbabilities[i]
					* CondNProbabilities[i][aces];
			if (Prob_in > Prob_out) {
				predictions[i] = true;
			} else if (Prob_in < Prob_out) {
				predictions[i] = false;
			} else {
				Random rnd = new Random();
				predictions[i] = (rnd.nextInt(2) == 1) ? true : false;
			}
			// ranking function
			confidences[i] = Prob_in / (Prob_in + Prob_out);
		}
		return (confidences);
	}

	private double[] getBugzieScores(Instance instance) {

		double[] scores = new double[numLabels];
		for (int i = 0; i < numLabels; i++) {
			scores[i] = 1;
		}
		for (int i = 0; i < featureIndices.length - topicNum - 2; i++) {
			Double termValue = Double.parseDouble(instance
					.toString(termIndex[i]));
			if (termValue > 0) {
				for (int j = 0; j < numLabels; j++) {
					boolean canCompute = false;
					for (int m = 0; m < termSelections; m++) {
						if (topTermsIndex[j][m] == i) {
							canCompute = true;
							break;
						}
					}
					if (canCompute) {
						scores[j] = (1 - termFixerScores[i][j]) * scores[j];
					}
				}
			}
		}

		for (int i = 0; i < numLabels; i++) {
			scores[i] = 1 - scores[i];
		}
		return scores;

	}

	private double[] getTopicScores(Instance instance) {
		double[] scores = new double[numLabels];

		for (int i = 0; i < numLabels; i++) {
			scores[i] = 1;
		}

		for (int i = 0; i < topicNum; i++) {
			Double termValue = Double.parseDouble(instance
					.toString(topicIndex[i]));
			if (termValue > 0) {
				for (int j = 0; j < numLabels; j++) {
					scores[j] = (1 - topicDiscusserScore[i][j] * termValue)
							* scores[j];
				}
			}
		}

		for (int i = 0; i < numLabels; i++) {
			scores[i] = 1 - scores[i];

		}

		return (scores);
	}

	private double getTopkHit(int[][] actual, int[][] predict) {
		double sumaccuracy = 0;
		for (int i = 0; i < sampleNum; i++) {

			int total = 0;
			int merge = 0;

			for (int k = 0; k < numLabels; k++) {
				if (actual[i][k] == 1) {
					total++;
				}
				if (actual[i][k] == 1 && predict[i][k] == 1) {
					merge++;
				}
			}

			if (total != 0) {
				sumaccuracy += (double) merge / (double) total;
			}
		}

		return sumaccuracy;

	}

	@Override
	protected MultiLabelOutput makePredictionInternal(Instance instance)
			throws Exception, InvalidDataException {
		double[] bugzieScores = getBugzieScores(instance);
		double[] topicScores = getTopicScores(instance);
		double[] mlknnScores = getMlkNNScores(instance);
		double[] proSc = new double[numLabels];
		double[] compSc = new double[numLabels];
		double[] scores = new double[numLabels];

		for (int i = 0; i < numLabels; i++) {
			int compValue = (int) instance.value(featureIndices.length - 1);

			int prodValue = (int) instance.value(featureIndices.length - 2);

			proSc[i] = productScores[prodValue][i];
			compSc[i] = compScores[compValue][i];
		}

		for (int i = 0; i < numLabels; i++) {

			scores[i] = beta1 * topicScores[i] + beta2 * bugzieScores[i]
					+ beta3 * compSc[i] + beta4 * proSc[i] + alpha
					* mlknnScores[i];

		}

		MultiLabelOutput mlo = new MultiLabelOutput(
				findtopkMatch(scores, topk), 0);
		return mlo;

	}

	private double[] findtopkMatch(double[] predictScore, int topk) {
		int numFixers = predictScore.length;
		double[] newCopy = new double[numFixers];
		double[] copyRes = new double[numFixers];

		for (int j = 0; j < numFixers; j++) {
			copyRes[j] = predictScore[j];
		}
		Arrays.sort(copyRes);
		int cur = 0;

		for (int p = numFixers; p > 0; p--) {

			newCopy[cur] = copyRes[p - 1];
			cur++;
		}

		int[] predict = new int[topk];

		for (int i = 0; i < topk; i++)

		{

			double pro = newCopy[i];

			for (int j = 0; j < numFixers; j++)

			{

				if (predictScore[j] == pro)

				{
					boolean canContain = true;
					for (int p = 0; p < i; p++) {
						if (predict[p] == j) {
							canContain = false;
							break;
						}
					}
					if (canContain) {

						predict[i] = j;
						break;
					}

				}

			}
		}

		boolean res = false;

		double[] finascores = new double[numFixers];
		for (int i = 0; i < numFixers; i++) {
			finascores[i] = -1;
		}

		for (int p = 0; p < topk; p++) {
			finascores[predict[p]] = predictScore[predict[p]];
		}

		return finascores;

	}

	private int[] findTermIndex(double[] predictScore, int topk) {
		int numFixers = predictScore.length;
		double[] newCopy = new double[numFixers];
		double[] copyRes = new double[numFixers];

		for (int j = 0; j < numFixers; j++) {
			copyRes[j] = predictScore[j];
		}
		Arrays.sort(copyRes);
		int cur = 0;

		for (int p = numFixers; p > 0; p--) {

			newCopy[cur] = copyRes[p - 1];
			cur++;
		}

		int[] predict = new int[topk];

		for (int i = 0; i < topk; i++)

		{

			double pro = newCopy[i];

			for (int j = 0; j < numFixers; j++)

			{

				if (predictScore[j] == pro)

				{
					boolean canContain = true;
					for (int p = 0; p < i; p++) {
						if (predict[p] == j) {
							canContain = false;
							break;
						}
					}
					if (canContain) {

						predict[i] = j;
						break;
					}

				}

			}
		}

		return predict;

	}

}
