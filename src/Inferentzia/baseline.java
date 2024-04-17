package Inferentzia;

import weka.core.converters.ConverterUtils;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Calendar;

public class baseline {
    public static void main(String[] args) throws Exception{

        //ARGUMENTUAK TXARTO SARTU DIRA
        if (args.length < 2) {
            System.out.println("Sartu argumentuak ondo: ");
            System.out.println("1. Datu sorta dakarren .arff fitxategiaren path-a.");
            System.out.println("2. Emaitzak gordetzeko fitxategiaren path-a.");
            return;
        }

        //ARGUMENTUAK ONDO SARTU DIRA
        if (args.length == 2){
            System.out.println("Sartutako argumentuak: ");
            for (int i = 0; i < args.length; i++) {
                System.out.println((i + 1) + ". Path-a: " + args[i]);
            }
        }

        //DATUAK KARGATU
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[0]);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes()-1);

        //INSTANTZIAK RANDOMIZATU
        Randomize filter_random = new Randomize();
        filter_random.setRandomSeed(1);
        filter_random.setInputFormat(data);
        Instances data_random = Filter.useFilter(data, filter_random);
        System.out.println("Instantzia kopurua: " + data_random.numInstances());

        //TEST ETA TRAIN
        RemovePercentage filter_remove = new RemovePercentage();
        filter_remove.setPercentage(70);
        filter_remove.setInvertSelection(false);
        filter_remove.setInputFormat(data_random);
        Instances test_instances = Filter.useFilter(data_random, filter_remove);
        System.out.println("Test-en instantzia kopurua: " + test_instances.numInstances());

        filter_remove.setInvertSelection(true);
        filter_remove.setInputFormat(data_random);
        Instances train_instances = Filter.useFilter(data_random, filter_remove);
        System.out.println("Train-en instantzia kopurua: " + train_instances.numInstances());

        test_instances.setClassIndex(test_instances.numAttributes()-1);
        train_instances.setClassIndex(train_instances.numAttributes()-1);

        //SAILKATZAILEA
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(train_instances);

        //EBALUATZAILEA
        Evaluation eval = new Evaluation(train_instances);
        eval.evaluateModel(nb, test_instances);

        //EMAITZEN FITXATEGIA LORTU
        fitxategiaSortu(eval, args, data_random);
    }

    private static void fitxategiaSortu(Evaluation eval, String[] args, Instances data) {
        try{
            FileWriter fitxategia = new FileWriter(args[1]);
            PrintWriter pw = new PrintWriter(fitxategia);
            String fecha = new SimpleDateFormat("yyyy/MM/dd HH/mm/ss").format(Calendar.getInstance().getTime());
            pw.println("Exekuzio data: " + fecha);
            pw.println("Jasotako path-ak: ");
            for (int i = 0; i < args.length; i++){
                pw.println((i+1) + ". Path-a: " + args[i]);
            }
            pw.println("Nahasmen matrizea: " + eval.toMatrixString());
            pw.println(eval.toClassDetailsString());
            pw.println(eval.toSummaryString());

            //System.out.println("Klase minoritarioa: ");
            int[] maiztasunak = data.attributeStats(data.numAttributes()-1).nominalCounts;
            int MaizMin = maiztasunak[0];
            int MaizMinPos = 0;
            int i = 0;
            while (i < maiztasunak.length) {
                if (maiztasunak[i] < MaizMin){
                    MaizMin = maiztasunak[i];
                    MaizMinPos = i;
                }
                i++;
            }
            double precision = eval.precision(MaizMinPos);
            double recall = eval.recall(MaizMinPos);
            double FScore = eval.fMeasure(MaizMinPos);
            pw.println("");
            pw.println("klase minoritarioa: " + data.classAttribute().value(MaizMinPos));
            pw.println("klase minoritarioaren precision: " + precision);
            pw.println("klase minoritarioaren recall: " + recall);
            pw.println("klase minoritarioaren FScore: " + FScore);
            //pw.println("Coinciden, está bien");
            pw.close();
        }
        catch (Exception e){
            e.printStackTrace();
        }
    }
}
