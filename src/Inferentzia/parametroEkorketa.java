package Inferentzia;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.BayesNetEstimator;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
//import weka.classifiers.bayes.net.estimate.;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.classifiers.bayes.net.search.local.*;

import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;

public class parametroEkorketa {
    public static void main(String[] args) throws Exception {

        // PARAMETRO EKORKETA

        //TODO
        /*
        QUE HAY QUE HACER:
        - LOS ALGORITMOS DE BUSQUEDA NO FUNCIONAN CON EL BAYESNETESTIMATOR PORQUE NO SON COMPATIBLES --> HABRÍA QUE EKORTUAR ALPHA
        - CON SIMPLEESTIMATOR LOS ALGORITMOS SI FUNCIONAN --> HABRÍA QUE BUSCAR EL MÁS OPTIMO
        */

        //1. Entrenamendurako datuak kargatu
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[0]);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes()-1);

        //2. Klase minoritarioa lortu
        int klaseMinoritarioaIndex = Utils.minIndex(data.attributeStats(data.classIndex()).nominalCounts);
        System.out.println("Klase minoritarioa: " + data.attribute(data.classIndex()).value(klaseMinoritarioaIndex));

        int klaseMayoritarioaIndex = Utils.maxIndex(data.attributeStats(data.classIndex()).nominalCounts);
        System.out.println("Klase mayoritarioa: " + data.attribute(data.classIndex()).value(klaseMayoritarioaIndex));

        //TODO
        //3. Bilaketa algoritmo desberdinak zehaztu
        K2 k2 = new K2();
        HillClimber hillClimber = new HillClimber();
        RepeatedHillClimber repeatedHillClimber = new RepeatedHillClimber();
        TAN tan = new TAN();
        SimulatedAnnealing simulatedAnnealing = new SimulatedAnnealing();
        TabuSearch tabuSearch = new TabuSearch();
        GeneticSearch geneticSearch = new GeneticSearch();
        //SearchAlgorithm[] algorithms = new SearchAlgorithm[]{k2, hillClimber, repeatedHillClimber, tan, simulatedAnnealing, tabuSearch, geneticSearch};
        SearchAlgorithm[] algorithms = new SearchAlgorithm[]{k2, hillClimber, repeatedHillClimber, tan, simulatedAnnealing, tabuSearch};

        //TODO
        //Estimadoreak zehaztu
        BayesNetEstimator bayesNetEstimator = new BayesNetEstimator();
        //bayesNetEstimator.setAlpha();
        SimpleEstimator simpleEstimator= new SimpleEstimator();
        //simpleEstimator.setAlpha();

        //4. Parametro optimoenak lortzeko aldagaiak
        SearchAlgorithm bestSearchAlgorithm = k2;
        //BayesNetEstimator bestEstimator = null;
        Double bestFmeasure = 0.0;
        Double bestAlpha = 0.0;
        Double bestPrecision = 0.0;

        int iterazioKop = 1;

        //TODO
        //5. Estimadore desberdinak
        //for(int i = 0; i < 2; i++){
        for(double alpha = 0.25; alpha < 1; alpha += 0.25){
            for (SearchAlgorithm algorithm : algorithms){
                System.out.println(iterazioKop + ". ITERAZIOA "+java.time.LocalDateTime.now().toString());

                //parametroak printeatu
                System.out.println("Alpha balioa: " + alpha);
                System.out.println("SearchAlgorithm: " + algorithm.getClass());
                //System.out.println("Estimator: SimpleEstimator" );
                System.out.println("Estimator: BayesNetEstimator" );

                //5.1. Sailkatzailea sortu
                BayesNet bayesNet = new BayesNet();
                simpleEstimator.setAlpha(alpha);
                bayesNet.setEstimator(simpleEstimator);

                //TODO
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
                bayesNet.setSearchAlgorithm(algorithm);

                //HOLD OUT
                //5.3. Instantziak nahastu, randomize
                Randomize filterRandomize = new Randomize();
                //filterRandomize.setRandomSeed(1);
                filterRandomize.setRandomSeed(iterazioKop);
                filterRandomize.setInputFormat(data);
                Instances dataRandom = Filter.useFilter(data, filterRandomize);
                dataRandom.setClassIndex(dataRandom.numAttributes()-1);

                /*
                //STRATIFIED HOLD OUT
                StratifiedRemoveFolds filter_remove = new StratifiedRemoveFolds();
                filter_remove.setNumFolds(5);
                filter_remove.setFold(1);
                filter_remove.setInvertSelection(false);
                filter_remove.setInputFormat(dataRandom);

                //TEST Y TRAIN
                Instances test = Filter.useFilter(dataRandom, filter_remove);
                test.setClassIndex(test.numAttributes()-1);
                System.out.println("Test-en instantzia kopurua: " + test.numInstances());

                filter_remove.setInvertSelection(true);
                filter_remove.setInputFormat(dataRandom);
                Instances train = Filter.useFilter(dataRandom, filter_remove);
                train.setClassIndex(train.numAttributes()-1);
                System.out.println("Train-en instantzia kopurua: " + train.numInstances());
                */

                //5.4. test multzoak lortu
                RemovePercentage filterRemove = new RemovePercentage();
                filterRemove.setPercentage(70);
                filterRemove.setInvertSelection(false);
                filterRemove.setInputFormat(dataRandom);
                Instances test = Filter.useFilter(dataRandom, filterRemove);
                test.setClassIndex(test.numAttributes()-1);
                System.out.println("Test-en instantzia kopurua: " + test.numInstances());

                //5.5. train multzoa lortu
                filterRemove.setInvertSelection(true);
                filterRemove.setInputFormat(dataRandom);
                Instances train = Filter.useFilter(dataRandom, filterRemove);
                train.setClassIndex(train.numAttributes()-1);
                System.out.println("Train-en instantzia kopurua: " + train.numInstances());

                //5.6. Sailkatzailea entrenatu
                bayesNet.buildClassifier(train);

                //5.7. Ebaluazioa egin
                Evaluation eval = new Evaluation(train);
                eval.evaluateModel(bayesNet, test);
                System.out.println(eval.toSummaryString());

                //5.8. Parametro optimoak lortu
                //double fMeasure = eval.fMeasure(klaseMinoritarioaIndex);
                //double fMeasure = eval.weightedFMeasure();
                //System.out.println("F-measure klase minoritarioarekiko: " + fMeasure);


                System.out.println(eval.precision(klaseMayoritarioaIndex));

                if (eval.precision(klaseMayoritarioaIndex) > bestPrecision){
                    bestPrecision = eval.precision(klaseMayoritarioaIndex);
                    bestSearchAlgorithm = algorithm;
                    bestAlpha = alpha;
                }

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
        System.out.println("Best precision: " + bestPrecision);
        System.out.println("Best alpha: " + bestAlpha);
        //System.out.println("Best FMeasure: " + bestFmeasure);
        System.out.println("Best SearchAlgorithm: " + bestSearchAlgorithm.getClass());

        //6. Parametro optimoak fitxategian gorde
        File file = new File(args[1]);
        FileWriter fw = new FileWriter(file);

        fw.write("Erabilitako estimadorea: SimpleEstimator\n");
        fw.write("SearchAlgorithm optimoa: " + bestSearchAlgorithm.getClass() + "\n");
        fw.write("Alpha optimoa: " + bestAlpha + "\n");
        fw.write("lortutako precision hoberena: " + bestPrecision + "\n");

        fw.flush();
        fw.close();
    }

}


