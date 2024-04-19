package Inferentzia;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.classifiers.bayes.net.search.local.*;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Random;

public class Inferentzia {
    public static void main(String[] args) throws Exception{
        if (args.length != 7){
            System.out.println("Helburua: Inferentzia --> parametroak.txt, Modeloa.model, kalitateEstimazioaModelo.txt, kalitateEstimazioaBaseline.txt lortzea.");
            System.out.println("Aurre-baldintzak: ");
            System.out.println("    Sartu beharreko argumentuak hurrengoak dira: ");
            System.out.println("    0. trainBOWFSS.arff fitxategiaren path-a.");
            System.out.println("    1. devBOWFSS.arff fitxategiaren path-a.");
            System.out.println("    2. parametroak.txt fitxategiaren path-a.");
            System.out.println("    3. traindev.arff fitxategiaren path-a.");
            System.out.println("    4. Modeloa.model fitxategiaren path-a.");
            System.out.println("    5. kalitateEstimazioaModelo.txt fitxategiaren path-a.");
            System.out.println("    6. kalitateEstimazioaBaseline.txt fitxategiaren path-a.");
            System.out.println("    java -jar getARFF.java \"/path/trainBOWFSS.arff\" \"/path/defBOWFSS.arff\" \"/path/parametroak.txt\" \"/path/traindev.arff\" \"/path/Modeloa.model\" \"/path/kalitateEstimazioaModelo.txt\" \"/path/kalitateEstimazioaBaseline.txt\"");
            System.out.println("Post-baldintzak: ");
            System.out.println("    parametroak.txt, Modeloa.model, kalitateEstimazioaModelo.txt, kalitateEstimazioaBaseline.txt lortzea.");
            System.out.println("    Bidean sortutako fitxategiak gordetzea ikusteko ondo goazen.");
        } else {
            String trainBOWFSSarff = args[0];
            String devBOWFSSarff = args[1];
            String parametroaktxt = args[2];
            String traindevarff = args[3];
            String modeloa = args[4];
            String kalitateEstimazioaModelotxt = args[5];
            String kalitateEstimazioaBaselinetxt = args[6];

            parametroEkorketa(trainBOWFSSarff,devBOWFSSarff,parametroaktxt);
            getModel(trainBOWFSSarff, devBOWFSSarff, traindevarff, modeloa, kalitateEstimazioaModelotxt, parametroaktxt);
            baseline(trainBOWFSSarff, devBOWFSSarff, traindevarff, kalitateEstimazioaBaselinetxt);
        }
    }

    private static void parametroEkorketa (String trainBOWFSSarff, String devBOWFSSarff, String parametroaktxt) throws Exception{
        // PARAMETRO EKORKETA
            /*
            QUE HAY QUE HACER?:
            - LOS ALGORITMOS DE BUSQUEDA NO FUNCIONAN CON EL BAYESNETESTIMATOR PORQUE NO SON COMPATIBLES --> DESCARTADO
            - CON SIMPLEESTIMATOR LOS ALGORITMOS SI FUNCIONAN --> HABRÍA QUE BUSCAR EL MÁS OPTIMO
            - BUSCAR EL ALPHA MÁS OPTIMO
            */

        //1. Entrenamendurako datuak kargatu
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(trainBOWFSSarff);
        Instances trainBOWFSS = source.getDataSet();
        trainBOWFSS.setClassIndex(trainBOWFSS.numAttributes() - 1);

        ConverterUtils.DataSource source2 = new ConverterUtils.DataSource(devBOWFSSarff);
        Instances devBOWFSS = source2.getDataSet();
        devBOWFSS.setClassIndex(devBOWFSS.numAttributes() - 1);

        //2. Klase minoritarioa/mayoritarioa lortu
        int klaseMinoritarioaIndex = Utils.minIndex(trainBOWFSS.attributeStats(trainBOWFSS.classIndex()).nominalCounts);
        System.out.println("Klase minoritarioa: " + trainBOWFSS.attribute(trainBOWFSS.classIndex()).value(klaseMinoritarioaIndex));

        int klaseMayoritarioaIndex = Utils.maxIndex(trainBOWFSS.attributeStats(trainBOWFSS.classIndex()).nominalCounts);
        System.out.println("Klase mayoritarioa: " + trainBOWFSS.attribute(trainBOWFSS.classIndex()).value(klaseMayoritarioaIndex));

        //3. Bilaketa algoritmo desberdinak zehaztu
        K2 k2 = new K2();
        HillClimber hillClimber = new HillClimber();
        RepeatedHillClimber repeatedHillClimber = new RepeatedHillClimber();
        TAN tan = new TAN();
        //SimulatedAnnealing simulatedAnnealing = new SimulatedAnnealing();
        TabuSearch tabuSearch = new TabuSearch();
        //GeneticSearch geneticSearch = new GeneticSearch();
        //SearchAlgorithm[] algorithms = new SearchAlgorithm[]{k2, hillClimber, repeatedHillClimber, tan, simulatedAnnealing, tabuSearch, geneticSearch};
        SearchAlgorithm[] algorithms = new SearchAlgorithm[]{k2, hillClimber, repeatedHillClimber, tan, tabuSearch};

        //Estimadoreak zehaztu
        //BayesNetEstimator bayesNetEstimator = new BayesNetEstimator();
        SimpleEstimator simpleEstimator = new SimpleEstimator();

        //4. Parametro optimoenak lortzeko aldagaiak
        SearchAlgorithm bestSearchAlgorithm = k2;
        Double bestFmeasure = 0.0;
        Double bestAlpha = 0.0;
        //Double bestPrecision = 0.0;

        int iterazioKop = 1;

        //5. Algoritmo desberdinak alpha desberdiñentzako
        //for(int i = 0; i < 2; i++){
        for (double alpha = 0.25; alpha < 1; alpha += 0.25) {
            for (SearchAlgorithm algorithm : algorithms) {
                System.out.println(iterazioKop + ". ITERAZIOA --------------------" + java.time.LocalDateTime.now().toString());

                //parametroak printeatu
                System.out.println("Alpha balioa: " + alpha);
                System.out.println("SearchAlgorithm: " + algorithm.getClass());
                System.out.println("Estimator: SimpleEstimator");
                //System.out.println("Estimator: BayesNetEstimator" );

                //5.1. Sailkatzailea sortu
                BayesNet bayesNet = new BayesNet();

                //5.2. Parametroak finkatu
                    /*
                    if (i == 0){
                        bayesNet.setOptions(new String[]{"-E", "weka.classifiers.bayes.net.estimate.BayesNetEstimator"});
                        bayesNet.setEstimator(bayesNetEstimator);
                    } else {
                        bayesNet.setOptions(new String[]{"-E", "weka.classifiers.bayes.net.estimate.SimpleEstimator"});
                        bayesNet.setEstimator(simpleEstimator);
                    }
                    */
                simpleEstimator.setAlpha(alpha);
                bayesNet.setEstimator(simpleEstimator);
                bayesNet.setSearchAlgorithm(algorithm);

                System.out.println("Train-en instantzia kopurua: " + trainBOWFSS.numInstances());
                System.out.println("dev-en instantzia kopurua: " + devBOWFSS.numInstances());

                //HOLD OUT
                //5.3. Sailkatzailea entrenatu
                bayesNet.buildClassifier(trainBOWFSS);

                //5.4. Ebaluazioa egin
                Evaluation eval = new Evaluation(trainBOWFSS);
                eval.evaluateModel(bayesNet, devBOWFSS);
                System.out.println(eval.toSummaryString());
                System.out.println(eval.toMatrixString());
                System.out.println(eval.toClassDetailsString());

                //5.5. Parametro optimoak lortu
                double fMeasure = eval.fMeasure(klaseMinoritarioaIndex);
                System.out.println("F-measure klase minoritarioarekiko: " + fMeasure);
                //System.out.println(eval.precision(klaseMayoritarioaIndex));

                if (fMeasure > bestFmeasure) {
                    bestFmeasure = fMeasure;
                    bestSearchAlgorithm = algorithm;
                    bestAlpha = alpha;
                }

                System.out.println("Best F-measure klase minoritarioarekiko: " + bestFmeasure);

                    /* Egindako beste proba batzuk (sin mas)
                    if (eval.precision(klaseMayoritarioaIndex) > bestPrecision){
                        bestPrecision = eval.precision(klaseMayoritarioaIndex);
                        bestSearchAlgorithm = algorithm;
                        bestAlpha = alpha;
                    }
                    */
                    /*
                    System.out.println(eval.fMeasure(klaseMinoritarioaIndex));

                    if (eval.fMeasure(klaseMinoritarioaIndex) > bestPrecision){
                        bestFmeasure = eval.fMeasure(klaseMinoritarioaIndex);
                        bestSearchAlgorithm = algorithm;
                        bestAlpha = alpha;
                    }
                    */
                    /*
                    if (fMeasure > bestFmeasure){
                        bestFmeasure = fMeasure;
                        bestSearchAlgorithm = algorithm;
                        bestAlpha = alpha;
                        if (i == 0){
                            bestEstimator = bayesNetEstimator;
                        } else {
                            bestEstimator = simpleEstimator;
                        }
                    }
                    */
                iterazioKop++;
            }
        }
        //System.out.println("Best precision: " + bestPrecision);
        System.out.println("Best alpha: " + bestAlpha);
        System.out.println("Best FMeasure: " + bestFmeasure);
        System.out.println("Best SearchAlgorithm: " + bestSearchAlgorithm.getClass());

        //6. Parametro optimoak fitxategian gorde
        File file = new File(parametroaktxt);
        FileWriter fw = new FileWriter(file);

        fw.write("Erabilitako estimadorea: SimpleEstimator\n");
        fw.write("SearchAlgorithm optimoa: " + bestSearchAlgorithm.getClass() + "\n");
        fw.write("Alpha optimoa: " + bestAlpha + "\n");
        fw.write("lortutako fmeasure hoberena: " + bestFmeasure + "\n");
        //fw.write("lortutako precision hoberena: " + bestPrecision + "\n");

        fw.flush();
        fw.close();
    }

    private static void getModel (String trainBOWFSSarff, String devBOWFSSarff, String traindevarff, String modeloa, String kalitateEstimazioaModelotxt, String parametroaktxt) throws Exception{
        //Aquí hay que juntar train y dev para hacer el buildclassifier

        // Train datuak kargatu
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(trainBOWFSSarff);
        Instances train = source.getDataSet();
        train.setClassIndex(train.numAttributes() - 1);

        // dev datuak kargatu
        ConverterUtils.DataSource source2 = new ConverterUtils.DataSource(devBOWFSSarff);
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
        saver.setFile(new File(traindevarff));
        saver.writeBatch();

        //1. Argumentuak gorde
        ConverterUtils.DataSource source3 = new ConverterUtils.DataSource(traindevarff);
        Instances train_dev = source3.getDataSet();
        train_dev.setClassIndex(train_dev.numAttributes() - 1);
        String pathModel = modeloa;
        String emaitzakPath = kalitateEstimazioaModelotxt;
        File parametroOptimoak = new File(parametroaktxt);

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

    private static void baseline (String trainBOWFSSarff, String devBOWFSSarff, String traindevarff, String kalitateEstimazioaBaselinetxt) throws Exception{
        //1. Datuak kargatu
        // train kargatu
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(trainBOWFSSarff);
        Instances train = source.getDataSet();
        train.setClassIndex(train.numAttributes() - 1);

        // dev kargatu
        ConverterUtils.DataSource source2 = new ConverterUtils.DataSource(devBOWFSSarff);
        Instances dev = source2.getDataSet();
        dev.setClassIndex(dev.numAttributes() - 1);

        // traindev kargatu
        ConverterUtils.DataSource source3 = new ConverterUtils.DataSource(traindevarff);
        Instances traindev = source3.getDataSet();
        traindev.setClassIndex(traindev.numAttributes() - 1);

        String emaitzakPath = kalitateEstimazioaBaselinetxt;

        //2. Modeloa sortu
        NaiveBayes naiveBayes = new NaiveBayes();
        naiveBayes.buildClassifier(traindev);

        //3. Kalitatearen estimazioa
        System.out.println("Ebaluazioa egiten...");
        File emaitzak = new File(emaitzakPath);
        FileWriter fw = new FileWriter(emaitzak);
        fw.write("################## KALITATEAREN ESTIMAZIOA ##################\n\n\n");

        //EZ-ZINTZOA
        fw.write("-----------EZ ZINTZOA------------\n\n");
        System.out.println("Ebaluazio EZ-ZINTZOA...");
        Evaluation evalEZintzoa = new Evaluation(traindev);
        evalEZintzoa.evaluateModel(naiveBayes, traindev);
        fw.write("\n" + evalEZintzoa.toClassDetailsString() + "\n");
        fw.write("\n" + evalEZintzoa.toSummaryString() + "\n");
        fw.write("\n" + evalEZintzoa.toMatrixString() + "\n");
        System.out.println("Ebaluazio EZ-ZINTZOA eginda");

        //CROSS VALIDATION
        fw.write("-----------CROSS VALIDATION----------\n\n");
        System.out.println("10 FOLD CROSS VALIDATION ebaluazioa...");
        Evaluation eval10fCV = new Evaluation(traindev);
        eval10fCV.crossValidateModel(naiveBayes, traindev, 10, new Random(1));
        fw.write("\n" + eval10fCV.toClassDetailsString() + "\n");
        fw.write("\n" + eval10fCV.toSummaryString() + "\n");
        fw.write("\n" + eval10fCV.toMatrixString() + "\n");
        System.out.println("10 FOLD CROSS VALIDATION ebaluazioa eginda");

        //HOLD OUT
        System.out.println("HOLD OUT ebaluazioa...");
        // Sailkatzailea entrenatu --> train
        NaiveBayes naiveBayes2 = new NaiveBayes();
        naiveBayes2.buildClassifier(train);

        // Ebaluazioa egin --> dev
        Evaluation evalHO = new Evaluation(train);
        evalHO.evaluateModel(naiveBayes2, dev);
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
