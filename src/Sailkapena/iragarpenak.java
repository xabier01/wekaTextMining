package Sailkapena;

import weka.classifiers.bayes.BayesNet;

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

        if (args.length != 7){
            System.out.println("Helburua: test_blind.txt fitxategiko mezuen iragarpen/sailkapena egitea.");
            System.out.println("          (lehen bezala, test_blind.txt garbitu, .arff-ra egokitu eta hiztegiarekin konpatiblea egin)");
            System.out.println("Aurre-baldintzak: ");
            System.out.println("    Sartu beharreko argumentuak hurrengoak dira: ");
            System.out.println("    0. Modelo.model artxiboaren path-a.");
            System.out.println("    1. test_blind.txt fitxategiaren path-a.");
            System.out.println("    2. test_blindGarbia.txt fitxategiaren path-a");
            System.out.println("    3. test_blind.arff fitxategiaren path-a.");
            System.out.println("    4. hiztegiaBOWFSS.txt fitxategiaren path-a");
            System.out.println("    5. predictions.txt fitxategiaren path-a.");
            System.out.println("    6. testBlindFinal.arff fitxategiaren path-a (ikusteko bakarrik)");
            System.out.println("    java -jar iragarpenak.java \"/path/Modeloa.model\" \"/path/test_blind.txt\" \"/path/test_blindGarbia.txt\" \"/path/test_blind.arff\" \"/path/hiztegiaBOWFSS.txt\" \"/path/predictions.txt\" \"/path/testBlindFinal.arff\"");
            System.out.println("Post-baldintzak: ");
            System.out.println("    Iragarpenak gordetzea.");
        } else {
            karaktereArraroakKendu(args[1], args[2]);

            BufferedReader reader = new BufferedReader(new FileReader(args[2]));
            BufferedWriter writer = new BufferedWriter(new FileWriter(args[3]));

            // ARFF-aren goiburua idatzi
            writer.write("@relation spam\n\n");
            writer.write("@attribute message string\n\n");
            writer.write("@attribute klasea {spam, ham}\n");
            writer.write("@data\n");

            String line;
            int lerroa = 1;
            while ((line = reader.readLine()) != null) {
                if (args[2].contains("test")) {

                    // lerroa ARFF fitxategian idatzi
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
            testBlind.setClassIndex(testBlind.numAttributes() - 1);

            FixedDictionaryStringToWordVector fds = new FixedDictionaryStringToWordVector();
            File hiztegiBerria = new File(args[4]);
            fds.setDictionaryFile(hiztegiBerria);
            fds.setInputFormat(testBlind);

            testBlind = Filter.useFilter(testBlind, fds);
            System.out.println("testBlind-en atributu kop: " + (testBlind.numAttributes() - 1));

            SparseToNonSparse filter = new SparseToNonSparse();
            filter.setInputFormat(testBlind);
            testBlind = Filter.useFilter(testBlind, filter);

            Reorder reorder = new Reorder();
            reorder.setAttributeIndices("2-" + testBlind.numAttributes() + ",1");
            reorder.setInputFormat(testBlind);
            testBlind = Filter.useFilter(testBlind, reorder);

            //Esto es lo de testBlindFinal.arff, que lo guardo para mirarlo en weka (pero no hace falta guardarlo)
            ArffSaver saver = new ArffSaver();
            saver.setInstances(testBlind);
            saver.setFile(new File(args[6]));
            saver.writeBatch();

            BayesNet model = (BayesNet) SerializationHelper.read(args[0]);

            // IRAGARPENAK
            testBlind.setClassIndex(testBlind.numAttributes() - 1);
            FileWriter f2 = new FileWriter(args[5]);
            PrintWriter pw2 = new PrintWriter(f2);
            pw2.println("INSTANTZIA------IRAGARPENA");
            for (int i = 0; i < testBlind.numInstances(); i++) {
                double pred = model.classifyInstance(testBlind.instance(i));
                pw2.println("     " + (i + 1) + "            " + testBlind.classAttribute().value((int) pred));
            }
            pw2.close();
            System.out.println("Iragarpenak bukatu dira.");
        }
    }

    private static void karaktereArraroakKendu(String fileName, String fileResult) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(fileName));
        PrintWriter pw = new PrintWriter(fileResult);
        String line;

        while ((line = br.readLine()) != null) {
            String regex = "[^a-zA-Z0-9\\s]";
            line = line.replaceAll(regex, "");
            pw.println(line);
        }
        br.close();
        pw.close();
    }

}
