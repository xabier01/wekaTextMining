package Inferentzia;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.classifiers.bayes.net.search.local.*;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.*;
import java.util.Random;

public class getModel {
    public static void main(String[] args) throws Exception {
        //TODO
        //Aquí hay que juntar train y dev para hacer el buildclassifier

        // Cargar conjunto de datos de entrenamiento
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[0]);
        Instances train = source.getDataSet();
        train.setClassIndex(train.numAttributes()-1);

        // Cargar conjunto de datos de desarrollo
        ConverterUtils.DataSource source2 = new ConverterUtils.DataSource(args[1]);
        Instances dev = source2.getDataSet();
        dev.setClassIndex(dev.numAttributes()-1);

        // Combinar los conjuntos de datos
        Instances combined = new Instances(train);
        for (int i = 0; i < dev.numInstances(); i++) {
            combined.add(dev.instance(i));
        }

        // Guardar el conjunto de datos combinado en un nuevo archivo ARFF
        ArffSaver saver = new ArffSaver();
        saver.setInstances(combined);
        saver.setFile(new File(args[2]));
        saver.writeBatch();

        //1. Argumentuak gorde
        ConverterUtils.DataSource source3 = new ConverterUtils.DataSource(args[2]);
        Instances train_dev = source3.getDataSet();
        train_dev.setClassIndex(train_dev.numAttributes()-1);
        String pathModel = args[3];
        String emaitzakPath = args[4];
        File parametroOptimoak = new File(args[5]);

        //2. Parametroak lortu fitxategitik
        SearchAlgorithm searchAlgorithm = null;
        double alpha = 0.0;
        try (BufferedReader br = new BufferedReader(new FileReader(parametroOptimoak))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (line.contains("SearchAlgorithm optimoa")){
                    String Algoritmoa = line.split(":")[1];
                    if (Algoritmoa.contains("class weka.classifiers.bayes.net.search.local.K2")){
                        K2 k2 = new K2();
                        searchAlgorithm = k2;
                    } else if (Algoritmoa.contains("class weka.classifiers.bayes.net.search.local.HillClimber")) {
                        HillClimber hillClimber = new HillClimber();
                        searchAlgorithm = hillClimber;
                    } else if (Algoritmoa.contains("class weka.classifiers.bayes.net.search.local.RepeatedHillClimber")) {
                        RepeatedHillClimber repeatedHillClimber = new RepeatedHillClimber();
                        searchAlgorithm = repeatedHillClimber;
                    } else if (Algoritmoa.contains("class weka.classifiers.bayes.net.search.local.TAN")) {
                        TAN tan = new TAN();
                        searchAlgorithm = tan;
                    /*} else if (Algoritmoa.contains("class weka.classifiers.bayes.net.search.local.SimulatedAnnealing")) {
                        SimulatedAnnealing simulatedAnnealing = new SimulatedAnnealing();
                        searchAlgorithm = simulatedAnnealing;*/
                    } else if (Algoritmoa.contains("class weka.classifiers.bayes.net.search.local.TabuSearch")) {
                        TabuSearch tabuSearch = new TabuSearch();
                        searchAlgorithm = tabuSearch;
                    }

                    /*
                    if (Algoritmoa.contains("class weka.classifiers.bayes.net.search.local.")){
                        String AlgoritmoIzena = Algoritmoa.split(".")[2];
                        System.out.println("Algoritmoa: " + AlgoritmoIzena);
                        if (AlgoritmoIzena.equals("K2")){
                            K2 k2 = new K2();
                            searchAlgorithm = k2;
                        } else if (AlgoritmoIzena.equals("HillClimber")) {
                            HillClimber hillClimber = new HillClimber();
                            searchAlgorithm = hillClimber;
                        } else if (AlgoritmoIzena.equals("RepeatedHillClimber")) {
                            RepeatedHillClimber repeatedHillClimber = new RepeatedHillClimber();
                            searchAlgorithm = repeatedHillClimber;
                        } else if (AlgoritmoIzena.equals("TAN")) {
                            TAN tan = new TAN();
                            searchAlgorithm = tan;
                        } else if (AlgoritmoIzena.equals("SimulatedAnnealing")) {
                            SimulatedAnnealing simulatedAnnealing = new SimulatedAnnealing();
                            searchAlgorithm = simulatedAnnealing;
                        } else if (AlgoritmoIzena.equals("TabuSearch")) {
                            TabuSearch tabuSearch = new TabuSearch();
                            searchAlgorithm = tabuSearch;
                        }
                    }
                    */

                }else if (line.contains("Alpha optimoa")){
                    alpha = Double.parseDouble((line.split(":")[1]));
                }
            }
        }
        System.out.println("train_dev-en instantzia kopurua: " + train_dev.numInstances());
        System.out.println("train_dev-en atributu kopurua: " + train_dev.numAttributes());


        /*
        //Hiztegi definitiboa
        FileWriter fWriter = new FileWriter(args[7]);
        try
        {
            File file=new File(args[6]);    //creates a new file instance
            FileReader fr=new FileReader(file);   //reads the file
            BufferedReader br=new BufferedReader(fr);  //creates a buffering character input stream
            StringBuffer sb=new StringBuffer();    //constructs a string buffer with no characters
            String line;
            br.readLine();
            for (int i = 0; i < train_dev.numAttributes()-1; i++) {
                String att = train_dev.attribute(i).name();
                while((line=br.readLine())!=null) {
                    String lineS = line.split(",")[0];
                    if(lineS.equals(train_dev.attribute(i).name())) {
                        fWriter.write(line + "\n");
                    }
                }
                fr = new FileReader(file);
                br = new BufferedReader(fr);
            }
        }
        catch(IOException e)
        {
            e.printStackTrace();
        }
        fWriter.close();

        FixedDictionaryStringToWordVector fds = new FixedDictionaryStringToWordVector();
        File hiztegiBerria = new File(args[7]);
        fds.setDictionaryFile(hiztegiBerria);
        fds.setInputFormat(train_dev);

        train_dev = Filter.useFilter(train_dev, fds);
        train_dev.setClassIndex(train_dev.numAttributes()-1);
        */

        //3. Modeloa sortu
        System.out.println("Modeloa sortzen...");
        System.out.println("SearchAlgorithm: " + searchAlgorithm.getClass());
        System.out.println("Alpha: " + alpha);

        BayesNet bayesNet = new BayesNet();
        SimpleEstimator simpleEstimator = new SimpleEstimator();
        simpleEstimator.setAlpha(alpha);
        bayesNet.setEstimator(simpleEstimator);
        bayesNet.setSearchAlgorithm(searchAlgorithm);
        bayesNet.buildClassifier(train_dev);

        //4. Modeloa gorde
        SerializationHelper.write(pathModel, bayesNet);

        //5. Ebaluatu eta estimazioa fitxategian gorde
        System.out.println("Ebaluazioa egiten...");
        File emaitzak = new File(emaitzakPath);
        FileWriter fw = new FileWriter(emaitzak);
        fw.write("################## KALITATEAREN ESTIMAZIOA ##################\n\n\n");

        //EZ-ZINTZOA
        fw.write("-----------EZ ZINTZOA------------\n\n");
        System.out.println("Ebaluazio EZ-ZINTZOA hasten...");
        Evaluation evalEZintzoa = new Evaluation(train_dev);
        evalEZintzoa.evaluateModel(bayesNet, train_dev);
        fw.write("\n" + evalEZintzoa.toClassDetailsString() + "\n");
        fw.write("\n" + evalEZintzoa.toSummaryString() + "\n");
        fw.write("\n" + evalEZintzoa.toMatrixString() + "\n");
        System.out.println("Ebaluazio EZ-ZINTZOA eginda...");

        //CROSS VALIDATION
        fw.write("-----------CROSS VALIDATION----------\n\n");
        System.out.println("10 FOLD CROSS VALIDATION ebaluazioa hasten...");
        Evaluation eval10fCV = new Evaluation(train_dev);
        eval10fCV.crossValidateModel(bayesNet, train_dev, 10, new Random(1));
        fw.write("\n" + eval10fCV.toClassDetailsString() + "\n");
        fw.write("\n" + eval10fCV.toSummaryString() + "\n");
        fw.write("\n" + eval10fCV.toMatrixString() + "\n");
        System.out.println("10 FOLD CROSS VALIDATION ebaluazioa eginda");

        //HOLD OUT
        System.out.println("HOLD OUT ebaluazioa hasten...");
        //5.6. Sailkatzailea entrenatu --> train
        BayesNet bayesNet2 = new BayesNet();
        SimpleEstimator simpleEstimator2 = new SimpleEstimator();
        simpleEstimator2.setAlpha(alpha);
        bayesNet2.setEstimator(simpleEstimator2);
        bayesNet2.setSearchAlgorithm(searchAlgorithm);
        bayesNet2.buildClassifier(train);

        //5.7. Ebaluazioa egin --> dev
        Evaluation evalHO = new Evaluation(train);
        evalHO.evaluateModel(bayesNet2, dev);
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
        Evaluation evalHoldOut = new Evaluation(train_dev);
        for(int i=0; i<20; i++) {
            //5 iterazio behin printeatu
            if (i < 5){
                System.out.println((i+1) + "/20 iterazioa");
            }

            //Randomize
            Randomize filter = new Randomize();
            filter.setInputFormat(train_dev);
            filter.setRandomSeed(i);
            Instances randomData = Filter.useFilter(train_dev,filter);
            randomData.setClassIndex(randomData.numAttributes()-1);

            //RemovePercentage --> train eta test lortu
            RemovePercentage filterRemove = new RemovePercentage();
            filterRemove.setInputFormat(randomData);
            filterRemove.setPercentage(70);
            filterRemove.setInvertSelection(false);
            Instances testHO = Filter.useFilter(randomData,filterRemove);
            testHO.setClassIndex(testHO.numAttributes()-1);
            System.out.println("TestHO-ren instantzia kopurua: " + testHO.numInstances());

            filterRemove.setInvertSelection(true);
            filterRemove.setInputFormat(randomData);
            Instances trainHO = Filter.useFilter(randomData, filterRemove);
            System.out.println("TrainHO-ren instantzia kopurua: " + trainHO.numInstances());

            //Ebaluatu
            evalHoldOut.evaluateModel(bayesNet, testHO);
        }
        */
    }
}
