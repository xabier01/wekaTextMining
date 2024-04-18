package Sailkapena;

import weka.classifiers.bayes.BayesNet;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.Prediction;

import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;

import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.instance.SparseToNonSparse;

import java.io.*;

public class iragarpenak {
    public static void main(String[] args) throws Exception {
		/*Argumentuak:
		 0- .model
		 1- testBlind.txt
		 2- testBlindGarbia.txt
		 3- testBlind.arff
		 4- hiztegiaBOWFSS
		 5- predictions.txt
		 6- testblindfinal.arff
		 3- HiztegiaFSS
		 4- testBlind.arff
		 5- testBlindGarbia.csv
		*/

        karaktereArraroakKendu(args[1], args[2]);

        BufferedReader reader = new BufferedReader(new FileReader(args[2]));
        BufferedWriter writer = new BufferedWriter(new FileWriter(args[3]));

        // Escribir encabezado del archivo ARFF
        writer.write("@relation spam\n\n");
        writer.write("@attribute message string\n\n");
        writer.write("@attribute klasea {spam, ham}\n");
        writer.write("@data\n");

        String line;
        int lerroa = 1;
        while ((line = reader.readLine()) != null) {
            if (args[2].contains("test")) {

                // Escribir la línea en el archivo ARFF
                writer.write("\"" + line + "\", ?\n");
                //writer.write("\"" + line + "\", unknown\n");
                System.out.println(lerroa + " " + line);
            }
            lerroa++;
        }
        reader.close();
        writer.close();

        ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[3]);
        Instances testBlind = source.getDataSet();
        testBlind.setClassIndex(testBlind.numAttributes()-1);

        /*
        NominalToString nts = new NominalToString();
        nts.setAttributeIndexes("6");
        nts.setInputFormat(testBlind);
        testBlind = Filter.useFilter(testBlind, nts);

        RenameAttribute ra = new RenameAttribute();
        ra.setAttributeIndices("2");
        ra.setReplace("moduleAttr");
        ra.setInputFormat(testBlind);
        testBlind = Filter.useFilter(testBlind, ra);

        ra = new RenameAttribute();
        ra.setAttributeIndices("3");
        ra.setReplace("ageAttr");
        ra.setInputFormat(testBlind);
        testBlind = Filter.useFilter(testBlind, ra);

        ra = new RenameAttribute();
        ra.setAttributeIndices("4");
        ra.setReplace("siteAttr");
        ra.setInputFormat(testBlind);
        testBlind = Filter.useFilter(testBlind, ra);

        ra = new RenameAttribute();
        ra.setAttributeIndices("5");
        ra.setReplace("sexAttr");
        ra.setInputFormat(testBlind);
        testBlind = Filter.useFilter(testBlind, ra);
        */

        FixedDictionaryStringToWordVector fds = new FixedDictionaryStringToWordVector();
        File hiztegiBerria = new File(args[4]);
        fds.setDictionaryFile(hiztegiBerria);
        fds.setInputFormat(testBlind);

        testBlind = Filter.useFilter(testBlind, fds);
        System.out.println("testBlind-en atributu kop: " + (testBlind.numAttributes()-1));

        SparseToNonSparse filter= new SparseToNonSparse();
        filter.setInputFormat(testBlind);
        testBlind = Filter.useFilter(testBlind, filter);

        Reorder reorder = new Reorder();
        reorder.setAttributeIndices("2-" + testBlind.numAttributes() + ",1");
        reorder.setInputFormat(testBlind);
        testBlind = Filter.useFilter(testBlind, reorder);

        ArffSaver saver = new ArffSaver();
        saver.setInstances(testBlind);
        saver.setFile(new File(args[6]));
        saver.writeBatch();

        //SMO model = (SMO) SerializationHelper.read(args[0]);
        BayesNet model = (BayesNet) SerializationHelper.read(args[0]);


        //IRAGARPENAK1
        Evaluation eval = new Evaluation(testBlind);
        eval.evaluateModel(model, testBlind);

        FileWriter fw = new FileWriter(args[7]);
        fw.write("BayesNet-ren iragarpenak: ");
        int instantzia = 1;
        for (Prediction p:eval.predictions()) {
            fw.write("\n"+instantzia+" iragarpena: "+p.predicted());
            instantzia++;
        }

        fw.close();

        // IRAGARPENAK2
        testBlind.setClassIndex(testBlind.numAttributes()-1);
        FileWriter f2 = new FileWriter(args[5]);
        PrintWriter pw2 = new PrintWriter(f2);
        pw2.println("INSTANTZIA------IRAGARPENA");
        for (int i = 0; i < testBlind.numInstances(); i++){
            double pred = model.classifyInstance(testBlind.instance(i));
            pw2.println("     " + (i+1) + "            " + testBlind.classAttribute().value((int)pred));
        }
        pw2.close();
        System.out.println("Iragarpenak bukatu dira.");
    }

    private static void karaktereArraroakKendu(String fileName, String fileResult) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(fileName));
        PrintWriter pw = new PrintWriter(fileResult);
        String line;

        while ((line = br.readLine()) != null) {
            // line = line.replace(subString, "");
            //line = line.replaceAll("[`'?.]", "");
            //pw.println(line);
            String regex = "[^a-zA-Z0-9\\s]";
            line = line.replaceAll(regex, "");
            pw.println(line);
        }
        br.close();
        pw.close();
    }

}
