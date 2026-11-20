package bug.developer.recommend.evaluation;

import java.util.ArrayList;
import java.util.List;

import com.ml.eval.NewEval;


import bug.developer.recommend.methods.DevRec;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.measure.ExampleBasedRecall;
import mulan.evaluation.measure.Measure;

public class progressValidation {

	public static String dirSouce = "gcc/";

	public static int fold = 10;
	public static int[] tops = new int[2];

	public static ExampleBasedRecall re = new ExampleBasedRecall(false);
	public static List<Measure> measureMent = new ArrayList<Measure>();

	public static void main(String[] args) throws Exception {

		tops[0] = 5;
		tops[1] = 10;

		measureMent.add(re);

		int numOfNeighbors = 15;

		int termSelections = 10;
		int percent = 5;
		{
			for (int p = 0; p < 2; p++) {

				evaluate(numOfNeighbors, termSelections, percent, p);

			}
		}

	}

	private static void evaluate(int numOfNeighbors, int termSelections,
			int percent, int p) throws InvalidDataFormatException, Exception {
		System.out.println("parameter Setting numberOfNeighbors=== "
				+ numOfNeighbors);
		System.out.println("parameter Setting termSelections=== "
				+ termSelections);

		System.out.println("parameter Setting topic percent=== " + percent);

		double topk = 0;
		int numberInstance = 0;

		for (int i = 0; i < fold; i++) {
			String originalWekaName = dirSouce + "fold" + i + ".arff";
			String labelWekaName = dirSouce + "fold" + i + ".xml";

			MultiLabelInstances originalWeka = new MultiLabelInstances(
					originalWekaName, labelWekaName);

			

			int wordNum = originalWeka.getFeatureIndices().length - 2;

			int topicNum = (int) (percent * wordNum / 100);

			String trainName = dirSouce + "fold" + i + "_topic" + percent
					+ "train.arff";

			String testName = dirSouce + "fold" + i + "_topic" + percent
					+ "test.arff";

			MultiLabelInstances trainSet = new MultiLabelInstances(trainName,
					labelWekaName);

			DevRec dm = new DevRec(numOfNeighbors, tops[p], tops[p], topicNum,
					termSelections);

			dm.build(trainSet);

			MultiLabelInstances testSet = new MultiLabelInstances(testName,
					labelWekaName);

			int testInstance = testSet.getDataSet().numInstances();

			NewEval eval = new NewEval();
			eval.setStrict(false);
			Evaluation e = eval.evaluate(dm, testSet, measureMent);
			String[] split1 = e.toString().split(":");
			System.out.println("current Fold  " + i);
			System.out.println(e.toString());

			topk += Double.parseDouble(split1[1]) * testInstance;
			numberInstance += testInstance;

		}

		System.out.println("Finally Top " + tops[p]);
		System.out.println("Recall@" + tops[p] + "=== " + (double) topk
				/ (double) numberInstance);
	}

	public static Instances removeProductandCom(MultiLabelInstances data)
			throws Exception {
		int[] features = new int[2];

		int[] featuresIndex = data.getFeatureIndices();
		features[0] = featuresIndex[featuresIndex.length - 1];
		features[1] = featuresIndex[featuresIndex.length - 2];
		Remove remove = new Remove();
		remove.setInvertSelection(false);
		remove.setAttributeIndicesArray(features);
		remove.setInputFormat(data.getDataSet());

		Instances subset = Filter.useFilter(data.getDataSet(), remove);

		return subset;
	}

}
