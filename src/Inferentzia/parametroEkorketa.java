package Inferentzia;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.BayesNetEstimator;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.classifiers.bayes.net.search.local.*;

import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;

import java.io.File;
import java.io.FileWriter;

public class parametroEkorketa {
    public static void main(String[] args) throws Exception {
        if (args.length != 3){
            System.out.println("Helburua: BayesNet modelo bat eraikitzeko parametro optimoak lortu eta gordetzea.");
            System.out.println("Aurre-baldintzak: ");
            System.out.println("    Sartu beharreko argumentuak hurrengoak dira: ");
            System.out.println("    0. trainBOWFSS.arff fitxategiaren path-a.");
            System.out.println("    1. devBoWFSS.arff fitxategiaren path-a.");
            System.out.println("    2. Parametroak.txt fitxategiaren path-a.");
            System.out.println("    java -jar getARFF.java \"/path/trainBOWFSS.arff\" \"/path/devBoWFSS.arff\" \"/path/Parametroak.txt\"");
            System.out.println("Post-baldintzak: ");
            System.out.println("    Parametroak.txt fitxategian parametro optimoak gordeta egotea.");
        } else {
            // PARAMETRO EKORKETA
            /*
            QUE HAY QUE HACER?:
            - LOS ALGORITMOS DE BUSQUEDA NO FUNCIONAN CON EL BAYESNETESTIMATOR PORQUE NO SON COMPATIBLES --> DESCARTADO
            - CON SIMPLEESTIMATOR LOS ALGORITMOS SI FUNCIONAN --> HABRÍA QUE BUSCAR EL MÁS OPTIMO
            - BUSCAR EL ALPHA MÁS OPTIMO
            */

            //1. Entrenamendurako datuak kargatu
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[0]);
            Instances trainBOWFSS = source.getDataSet();
            trainBOWFSS.setClassIndex(trainBOWFSS.numAttributes() - 1);

            ConverterUtils.DataSource source2 = new ConverterUtils.DataSource(args[1]);
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
            File file = new File(args[2]);
            FileWriter fw = new FileWriter(file);

            fw.write("Erabilitako estimadorea: SimpleEstimator\n");
            fw.write("SearchAlgorithm optimoa: " + bestSearchAlgorithm.getClass() + "\n");
            fw.write("Alpha optimoa: " + bestAlpha + "\n");
            fw.write("lortutako fmeasure hoberena: " + bestFmeasure + "\n");
            //fw.write("lortutako precision hoberena: " + bestPrecision + "\n");

            fw.flush();
            fw.close();
        }
    }

}


