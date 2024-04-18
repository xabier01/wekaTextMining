package Inferentzia;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.classifiers.bayes.net.search.local.*;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.classifiers.bayes.NaiveBayes;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Random;

public class baseline {
    public static void main(String[] args) throws Exception {

        //1. Argumentuak gorde
        // Cargar conjunto de datos de entrenamiento
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[0]);
        Instances train = source.getDataSet();
        train.setClassIndex(train.numAttributes()-1);

        // Cargar conjunto de datos de desarrollo
        ConverterUtils.DataSource source2 = new ConverterUtils.DataSource(args[1]);
        Instances dev = source2.getDataSet();
        dev.setClassIndex(dev.numAttributes()-1);

        // Cargar conjunto de datos completo
        ConverterUtils.DataSource source3 = new ConverterUtils.DataSource(args[2]);
        Instances traindev = source3.getDataSet();
        traindev.setClassIndex(traindev.numAttributes()-1);

        String emaitzakPath = args[3];

        //3. Modeloa sortu
        NaiveBayes naiveBayes = new NaiveBayes();
        naiveBayes.buildClassifier(traindev);

        //5. Ebaluatu eta estimazioa fitxategian gorde
        System.out.println("Ebaluazioa egiten...");
        File emaitzak = new File(emaitzakPath);
        FileWriter fw = new FileWriter(emaitzak);
        fw.write("################## KALITATEAREN ESTIMAZIOA ##################\n\n\n");

        //EZ-ZINTZOA
        fw.write("-----------EZ ZINTZOA------------\n\n");
        System.out.println("Ebaluazio EZ-ZINTZOA hasten...");
        Evaluation evalEZintzoa = new Evaluation(traindev);
        evalEZintzoa.evaluateModel(naiveBayes, traindev);
        fw.write("\n" + evalEZintzoa.toClassDetailsString() + "\n");
        fw.write("\n" + evalEZintzoa.toSummaryString() + "\n");
        fw.write("\n" + evalEZintzoa.toMatrixString() + "\n");
        System.out.println("Ebaluazio EZ-ZINTZOA eginda...");

        //CROSS VALIDATION
        fw.write("-----------CROSS VALIDATION----------\n\n");
        System.out.println("10 FOLD CROSS VALIDATION ebaluazioa hasten...");
        Evaluation eval10fCV = new Evaluation(traindev);
        eval10fCV.crossValidateModel(naiveBayes, traindev, 10, new Random(1));
        fw.write("\n" + eval10fCV.toClassDetailsString() + "\n");
        fw.write("\n" + eval10fCV.toSummaryString() + "\n");
        fw.write("\n" + eval10fCV.toMatrixString() + "\n");
        System.out.println("10 FOLD CROSS VALIDATION ebaluazioa eginda");

        //HOLD OUT
        System.out.println("HOLD OUT ebaluazioa hasten...");
        //5.6. Sailkatzailea entrenatu --> train
        NaiveBayes naiveBayes2 = new NaiveBayes();
        naiveBayes2.buildClassifier(train);

        //5.7. Ebaluazioa egin --> dev
        Evaluation evalHO = new Evaluation(train);
        evalHO.evaluateModel(naiveBayes2, dev);
        // System.out.println(evalHO.toSummaryString());
        // System.out.println(evalHO.toMatrixString());
        // System.out.println(evalHO.toClassDetailsString());

        fw.write("---------HOLDOUT---------\n\n");
        fw.write("\n" + evalHO.toClassDetailsString() + "\n");
        fw.write("\n" + evalHO.toSummaryString() + "\n");
        fw.write("\n" + evalHO.toMatrixString() + "\n");
        System.out.println("HOLD-OUT ebaluazioa eginda...");
        fw.close();

        /*
        //HOLD-OUT 20 ALDIZ
        fw.write("---------REPEATED HOLDOUT (20)---------\n\n");
        System.out.println("20 HOLD-OUT ebaluazioa hasten...");
        Evaluation evalHoldOut = new Evaluation(train);
        for (int i = 0; i < 20; i++) {
            //5 iterazio behin printeatu
            if (i < 5) {
                System.out.println("\t" + (i + 1) + "/20 iterazioa");
            }

            //Randomize
            Randomize filter = new Randomize();
            filter.setInputFormat(train);
            filter.setRandomSeed(i);
            Instances randomData = Filter.useFilter(train, filter);
            randomData.setClassIndex(randomData.numAttributes() - 1);

            //RemovePercentage --> train eta test lortu
            RemovePercentage filterRemove = new RemovePercentage();
            filterRemove.setInputFormat(randomData);
            filterRemove.setPercentage(70);
            filterRemove.setInvertSelection(false);
            Instances testHO = Filter.useFilter(randomData, filterRemove);
            testHO.setClassIndex(testHO.numAttributes() - 1);
            System.out.println("TestHO-ren instantzia kopurua: " + testHO.numInstances());

            filterRemove.setInvertSelection(true);
            filterRemove.setInputFormat(randomData);
            Instances trainHO = Filter.useFilter(randomData, filterRemove);
            System.out.println("TrainHO-ren instantzia kopurua: " + trainHO.numInstances());

            //Ebaluatu
            evalHoldOut.evaluateModel(naiveBayes, testHO);
        }
        */
    }
}
