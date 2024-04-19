package Inferentzia;

import weka.classifiers.Evaluation;
import weka.core.converters.ConverterUtils;
import weka.core.Instances;
import weka.classifiers.bayes.NaiveBayes;

import java.io.*;
import java.util.Random;

public class baseline {
    public static void main(String[] args) throws Exception {
        if (args.length != 4){
            System.out.println("Helburua: NaiveBayes eredua erabiliz behe bornea lortzea.");
            System.out.println("          Eredu sinple baten kalitatearen estimazioa lortzea.");
            System.out.println("Aurre-baldintzak: ");
            System.out.println("    Sartu beharreko argumentuak hurrengoak dira: ");
            System.out.println("    0. trainBOWFSS.arff fitxategiaren path-a.");
            System.out.println("    1. devBoWFSS.arff fitxategiaren path-a.");
            System.out.println("    2. traindev.arff fitxategiaren path-a.");
            System.out.println("    3. baseline.txt fitxategiaren path-a.");
            System.out.println("    java -jar getARFF.java \"/path/trainBOWFSS.arff\" \"/path/devBoWFSS.arff\" \"/path/traindev.arff\" \"/path/baseline.txt\"");
            System.out.println("Post-baldintzak: ");
            System.out.println("    Eredu sinple baten kalitatearen estimazioa gordetzea.");
        } else {
            //1. Datuak kargatu
            // train kargatu
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[0]);
            Instances train = source.getDataSet();
            train.setClassIndex(train.numAttributes() - 1);

            // dev kargatu
            ConverterUtils.DataSource source2 = new ConverterUtils.DataSource(args[1]);
            Instances dev = source2.getDataSet();
            dev.setClassIndex(dev.numAttributes() - 1);

            // traindev kargatu
            ConverterUtils.DataSource source3 = new ConverterUtils.DataSource(args[2]);
            Instances traindev = source3.getDataSet();
            traindev.setClassIndex(traindev.numAttributes() - 1);

            String emaitzakPath = args[3];

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
}
