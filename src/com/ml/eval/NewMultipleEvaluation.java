package com.ml.eval;

import java.util.ArrayList;
import java.util.HashMap;

import mulan.evaluation.Evaluation;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.Measure;

public class NewMultipleEvaluation extends MultipleEvaluation {

	private ArrayList<Evaluation> evaluations;
	private HashMap<String, Double> mean;
	private HashMap<String, Double> standardDeviation;

	/**
	 * Constructs a new object
	 */
	public NewMultipleEvaluation() {
		evaluations = new ArrayList<Evaluation>();
	}

	/**
	 * Constructs a new object with given array of evaluations and calculates
	 * statistics
	 * 
	 * @param someEvaluations
	 */
	public NewMultipleEvaluation(Evaluation[] someEvaluations) {
		evaluations = new ArrayList<Evaluation>();
		for (Evaluation e : someEvaluations) {
			evaluations.add(e);
		}
		calculateStatistics();
	}

	/**
	 * Computes mean and standard deviation of all evaluation measures
	 */
	public void calculateStatistics() {
		int size = evaluations.size();
		HashMap<String, Double> sums = new HashMap<String, Double>();

		// calculate sums of measures
		for (int i = 0; i < evaluations.size(); i++) {
			for (Measure m : evaluations.get(i).getMeasures()) {
				double value = Double.NaN;
				try {
					value = m.getValue();
				} catch (Exception ex) {
				}
				if (sums.containsKey(m.getName())) {
					sums.put(m.getName(), sums.get(m.getName()) + value);
				} else {
					sums.put(m.getName(), value);
				}
			}
		}
		mean = new HashMap<String, Double>();
		for (String measureName : sums.keySet()) {
			mean.put(measureName, sums.get(measureName) / size);
		}

		// calculate sums of squared differences from mean
		sums = new HashMap<String, Double>();

		for (int i = 0; i < evaluations.size(); i++) {
			for (Measure m : evaluations.get(i).getMeasures()) {
				double value = Double.NaN;
				try {
					value = m.getValue();
				} catch (Exception ex) {
				}
				if (sums.containsKey(m.getName())) {
					sums.put(m.getName(), sums.get(m.getName())
							+ Math.pow(value - mean.get(m.getName()), 2));
				} else {
					sums.put(m.getName(), Math.pow(value
							- mean.get(m.getName()), 2));
				}
			}
		}
		standardDeviation = new HashMap<String, Double>();
		for (String measureName : sums.keySet()) {
			standardDeviation.put(measureName, Math.sqrt(sums.get(measureName)
					/ size));
		}
	}

	/**
	 * Adds an evaluation results to the list of evaluations
	 * 
	 * @param evaluation
	 *            an evaluation result
	 */
	public void addEvaluation(Evaluation evaluation) {
		evaluations.add(evaluation);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		for (Measure m : evaluations.get(0).getMeasures()) {
			String measureName = m.getName();
			sb.append(measureName);
			sb.append(": ");
			sb.append(String.format("%.4f", mean.get(measureName)));
			sb.append("\u00B1");
			sb
					.append(String.format("%.4f", standardDeviation
							.get(measureName)));
			sb.append("\n");
		}
		return sb.toString();
	}

	public String getFolder() {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < evaluations.size(); i++) {
			
			sb.append(i);
			sb.append("  folder: ");
			sb.append("\n");
			for (Measure m : evaluations.get(i).getMeasures()) {
				String measureName = m.getName();					
				sb.append(measureName);
				sb.append(": ");
				double value = Double.NaN;
				try {
					value = m.getValue();
				} catch (Exception ex) {
				}
				
				sb.append(String.format("%.4f",value));
				sb.append("\n");
			}
		}
		return sb.toString();
	}

	/**
	 * Returns the mean value of a measure
	 * 
	 * @param measureName
	 *            the name of the measure
	 * @return the mean value of the measure
	 */
	public double getMean(String measureName) {
		return mean.get(measureName);
	}

	/**
	 * Returns a CSV string representation of the results
	 * 
	 * @return a CSV string representation of the results
	 */
	public String toCSV() {
		StringBuilder sb = new StringBuilder();
		for (Measure m : evaluations.get(0).getMeasures()) {
			String measureName = m.getName();
			sb.append(String.format("%.4f", mean.get(measureName)));
			sb.append("\u00B1");
			sb
					.append(String.format("%.4f", standardDeviation
							.get(measureName)));
			sb.append(";");
		}
		return sb.toString();
	}
}
