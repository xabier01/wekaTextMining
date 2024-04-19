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

import java.io.*;
import java.util.Random;

public class getModel {
    public static void main(String[] args) throws Exception {
        if (args.length != 6){
            System.out.println("Helburua: Lortutako parametro optimoekin BayesNet modeloa eraiki eta gordetzea.");
            System.out.println("          trainBOWFSS.arff eta devBoWFSS.arff fitxategiak elkatzea datu sorta osoa gordetzeko.");
            System.out.println("          Modelo honen kalitatearen estimazioa egin eta gordetzea.");
            System.out.println("Aurre-baldintzak: ");
            System.out.println("    Sartu beharreko argumentuak hurrengoak dira: ");
            System.out.println("    0. trainBOWFSS.arff fitxategiaren path-a.");
            System.out.println("    1. devBoWFSS.arff fitxategiaren path-a.");
            System.out.println("    2. traindev.arff fitxategiaren path-a (biak elkartzea eta gordetzea).");
            System.out.println("    3. Modeloa.model artxiboaren path-a.");
            System.out.println("    4. kalitateEstimazioa.txt fitxategiaren path-a.");
            System.out.println("    5. Parametroak.txt fitxategiaren path-a.");
            System.out.println("    java -jar getARFF.java \"/path/trainBOWFSS.arff\" \"/path/devBoWFSS.arff\" \"/path/traindev.arff\" \"/path/Modeloa.model\" \"/path/kalitateEstimazioa.txt\" \"/path/Parametroak.txt\"");
            System.out.println("Post-baldintzak: ");
            System.out.println("    Modelo optimoa gordetzea.");
            System.out.println("    traindev.arff fitxategia gordetzea (datu sorta osoa).");
            System.out.println("    Modeloaren kalitatearen estimazioa gordetzea.");
        } else {

            //Aquí hay que juntar train y dev para hacer el buildclassifier

            // Train datuak kargatu
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[0]);
            Instances train = source.getDataSet();
            train.setClassIndex(train.numAttributes() - 1);

            // dev datuak kargatu
            ConverterUtils.DataSource source2 = new ConverterUtils.DataSource(args[1]);
            Instances dev = source2.getDataSet();
            dev.setClassIndex(dev.numAttributes() - 1);

            // Biak batu
            Instances combined = new Instances(train);
            for (int i = 0; i < dev.numInstances(); i++) {
                combined.add(dev.instance(i));
            }

            // Bien batzea ARFF batean gorde
            ArffSaver saver = new ArffSaver();
            saver.setInstances(combined);
            saver.setFile(new File(args[2]));
            saver.writeBatch();

            //1. Argumentuak gorde
            ConverterUtils.DataSource source3 = new ConverterUtils.DataSource(args[2]);
            Instances train_dev = source3.getDataSet();
            train_dev.setClassIndex(train_dev.numAttributes() - 1);
            String pathModel = args[3];
            String emaitzakPath = args[4];
            File parametroOptimoak = new File(args[5]);

            //2. Parametroak lortu fitxategitik
            SearchAlgorithm searchAlgorithm = null;
            double alpha = 0.0;
            try (BufferedReader br = new BufferedReader(new FileReader(parametroOptimoak))) {
                String line;
                while ((line = br.readLine()) != null) {
                    if (line.contains("SearchAlgorithm optimoa")) {
                        String Algoritmoa = line.split(":")[1];
                        if (Algoritmoa.contains("class weka.classifiers.bayes.net.search.local.K2")) {
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
                    } else if (line.contains("Alpha optimoa")) {
                        alpha = Double.parseDouble((line.split(":")[1]));
                    }
                }
            }
            System.out.println("train_dev-en instantzia kopurua: " + train_dev.numInstances());
            System.out.println("train_dev-en atributu kopurua: " + train_dev.numAttributes());

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

            //5. Kalitatearen estimazioa
            System.out.println("Ebaluazioa egiten...");
            File emaitzak = new File(emaitzakPath);
            FileWriter fw = new FileWriter(emaitzak);
            fw.write("################## KALITATEAREN ESTIMAZIOA ##################\n\n\n");

            //EZ-ZINTZOA
            fw.write("-----------EZ ZINTZOA------------\n\n");
            System.out.println("Ebaluazio EZ-ZINTZOA...");
            Evaluation evalEZintzoa = new Evaluation(train_dev);
            evalEZintzoa.evaluateModel(bayesNet, train_dev);
            fw.write("\n" + evalEZintzoa.toClassDetailsString() + "\n");
            fw.write("\n" + evalEZintzoa.toSummaryString() + "\n");
            fw.write("\n" + evalEZintzoa.toMatrixString() + "\n");
            System.out.println("Ebaluazio EZ-ZINTZOA eginda");

            //CROSS VALIDATION
            fw.write("-----------CROSS VALIDATION----------\n\n");
            System.out.println("10 FOLD CROSS VALIDATION ebaluazioa...");
            Evaluation eval10fCV = new Evaluation(train_dev);
            eval10fCV.crossValidateModel(bayesNet, train_dev, 10, new Random(1));
            fw.write("\n" + eval10fCV.toClassDetailsString() + "\n");
            fw.write("\n" + eval10fCV.toSummaryString() + "\n");
            fw.write("\n" + eval10fCV.toMatrixString() + "\n");
            System.out.println("10 FOLD CROSS VALIDATION ebaluazioa eginda");

            //HOLD OUT
            System.out.println("HOLD OUT ebaluazioa...");
            // Sailkatzailea entrenatu --> train
            BayesNet bayesNet2 = new BayesNet();
            SimpleEstimator simpleEstimator2 = new SimpleEstimator();
            simpleEstimator2.setAlpha(alpha);
            bayesNet2.setEstimator(simpleEstimator2);
            bayesNet2.setSearchAlgorithm(searchAlgorithm);
            bayesNet2.buildClassifier(train);

            // Ebaluazioa egin --> dev
            Evaluation evalHO = new Evaluation(train);
            evalHO.evaluateModel(bayesNet2, dev);
            // System.out.println(evalHO.toSummaryString());
            // System.out.println(evalHO.toMatrixString());
            // System.out.println(evalHO.toClassDetailsString());

            fw.write("---------HOLDOUT---------\n\n");
            fw.write("\n" + evalHO.toClassDetailsString() + "\n");
            fw.write("\n" + evalHO.toSummaryString() + "\n");
            fw.write("\n" + evalHO.toMatrixString() + "\n");
            System.out.println("HOLD-OUT ebaluazioa eginda");
            fw.close();
        }
    }
}
