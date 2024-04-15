package Inferentzia;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.BayesNetEstimator;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.local.*;

import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.File;
import java.io.FileWriter;

public class parametroEkorketa {
    public static void main(String[] args) throws Exception {

        // PARAMETRO EKORKETA

        //1. Entrenamendurako datuak kargatu
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[0]);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes()-1);

        //2. Klase minoritarioa lortu
        int klaseMinoritarioaIndex = Utils.minIndex(data.attributeStats(data.classIndex()).nominalCounts);
        System.out.println("Klase minoritarioa: " + data.attribute(data.classIndex()).value(klaseMinoritarioaIndex));

        //TODO
        //3. Bilaketa algoritmo desberdinak zehaztu
        K2 k2 = new K2();
        HillClimber hillClimber = new HillClimber();
        RepeatedHillClimber repeatedHillClimber = new RepeatedHillClimber();
        TAN tan = new TAN();
        SimulatedAnnealing simulatedAnnealing = new SimulatedAnnealing();
        TabuSearch tabuSearch = new TabuSearch();
        GeneticSearch geneticSearch = new GeneticSearch();
        BayesNetSearchAlgorithm[] algorithms = new BayesNetSearchAlgorithm[]{k2, hillClimber, repeatedHillClimber, tan, simulatedAnnealing, tabuSearch, geneticSearch};

        //TODO
        //Estimadoreak zehaztu
        BayesNetEstimator bayesNetEstimator = new BayesNetEstimator();
        //bayesNetEstimator.
        SimpleEstimator simpleEstimator= new SimpleEstimator();

        //4. Parametro optimoenak lortzeko aldagaiak
        BayesNetSearchAlgorithm bestSearchAlgorithm = k2;
        BayesNetEstimator bestEstimator = null;
        Double bestFmeasure = 0.0;

        int iterazioKop = 1;

        //TODO
        //5. Estimadore desberdinak
        for(Estimadores){
            for (BayesNetSearchAlgorithm algorithm : algorithms){

                System.out.println("#############################################################################");
                System.out.println(iterazioKop + ". ITERAZIOA "+java.time.LocalDateTime.now().toString());

                //parametroak printeatu
                System.out.println("SearchAlgorithm: "  );
                System.out.println("Estimator: "  );

                //5.1. Sailkatzailea sortu
                BayesNet bayesNet = new BayesNet();

                //TODO
                //5.2. Parametroak finkatu
                /*bayesNet.setSearchAlgorithm();
                bayesNet.setEstimator();
                bayesNet.setOptions();*/

                //HOLD OUT
                //5.3. Instantziak nahastu, randomize
                Randomize filterRandomize = new Randomize();
                filterRandomize.setRandomSeed(iterazioKop);
                filterRandomize.setInputFormat(data);
                Instances dataRandom = Filter.useFilter(data, filterRandomize);

                //5.4. test multzoak lortu
                RemovePercentage filterRemove = new RemovePercentage();
                filterRemove.setPercentage(70);
                filterRemove.setInvertSelection(false);
                filterRemove.setInputFormat(dataRandom);
                Instances test = Filter.useFilter(dataRandom, filterRemove);
                System.out.println("Test-en instantzia kopurua: " + test.numInstances());

                //5.5. train multzoa lortu
                filterRemove.setInvertSelection(true);
                filterRemove.setInputFormat(dataRandom);
                Instances train = Filter.useFilter(dataRandom, filterRemove);
                System.out.println("Test-en instantzia kopurua: " + train.numInstances());

                //5.6. Sailkatzailea entrenatu
                bayesNet.buildClassifier(train);

                //5.7. Ebaluazioa egin
                Evaluation eval = new Evaluation(train);
                eval.evaluateModel(bayesNet, test);
                System.out.println(eval.toSummaryString());

                //5.8. Parametro optimoak lortu
                double fMeasure = eval.fMeasure(klaseMinoritarioaIndex);
                System.out.println("F-measure klase minoritarioarekiko: " + fMeasure);

                if (fMeasure > bestFmeasure){
                    bestFmeasure = fMeasure;
                    //bestSearchAlgorithm = ;
                    //bestEstimator = ;
                }
                iterazioKop++;
            }
        }

        //6. Parametro optimoak fitxategian gorde
        File file = new File(args[1]);
        FileWriter fw = new FileWriter(file);

        fw.write("SearchAlgorithm optimoa: " + bestSearchAlgorithm);
        fw.write("Best Learning Rate: " + bestEstimator);

        fw.flush();
        fw.close();
    }

}

/*
abstract class BayesNetSearchAlgorithm {
    public static final BayesNetSearchAlgorithm k2 = new K2();
    public static final BayesNetSearchAlgorithm hillClimber = new HillClimber();
    public static final BayesNetSearchAlgorithm repeatedHillClimber = new RepeatedHillClimber();
    public static final BayesNetSearchAlgorithm tan = new TAN();
    public static final BayesNetSearchAlgorithm simulatedAnnealing = new SimulatedAnnealing();
    public static final BayesNetSearchAlgorithm tabuSearch = new TabuSearch();
    public static final BayesNetSearchAlgorithm geneticSearch = new GeneticSearch();

    // Método para obtener una descripción del algoritmo
    public abstract String getDescription();

    // Método para obtener opciones específicas del algoritmo
    public abstract String[] getOptions();
}
*/
