package Aurreprozesamendua;

import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.*;

public class getARFF{
    public static void main(String[] args) throws Exception{
        try {
            if (args.length != 6){
                System.out.println("Helburua: train.txt eta dev.txt fitxategiak .arff fitxategi formatura egokitzea/pasatzea.");
                System.out.println("Aurre-baldintzak: ");
                System.out.println("    Sartu beharreko argumentuak hurrengoak dira: ");
                System.out.println("    0. train.txt fitxategiaren path-a.");
                System.out.println("    1. dev.txt fitxategiaren path-a.");
                System.out.println("    2. train.arff fitxategiaren path-a.");
                System.out.println("    3. dev.arff fitxategiaren path-a.");
                System.out.println("    4. trainGarbia.txt fitxategi garbiaren path-a (karaktere arraroak kenduta gordetzeko).");
                System.out.println("    5. devGarbia.txt fitxategi garbiaren path-a (karaktere arraroak kenduta gordetzeko).");
                System.out.println("    java -jar getARFF.java \"/path/train.txt\" \"/path/dev.txt\" \"/path/train.arff\" \"/path/dev.arff\" \"/path/trainGarbia.txt\" \"/path/devGarbia.txt\"");
                System.out.println("Post-baldintzak: ");
                System.out.println("    Sartutako train.arff fitxategian train.txt-ko datuak izatea formatu egokian.");
                System.out.println("    Sartutako dev.arff fitxategian dev.txt-ko datuak izatea formatu egokian.");
                System.out.println("    Egitura --> \"testua\", klasea --> egituradun .arff fitxategiak lortu.");
                System.out.println("    test (blind) txt fitxategiaren konbertsioa iragarpenak klasean egingo dugu.");
            } else {

                karaktereArraroakKendu(args[0], args[4]);
                karaktereArraroakKendu(args[1], args[5]);

                BufferedReader reader = new BufferedReader(new FileReader(args[4]));
                BufferedWriter writer = new BufferedWriter(new FileWriter(args[2]));

                // ARFF-aren goiburuak idatzi
                writer.write("@relation spam\n\n");
                writer.write("@attribute message string\n\n");
                writer.write("@attribute klasea {spam, ham}\n");
                writer.write("@data\n");

                String line;
                while ((line = reader.readLine()) != null) {
                    // Lerroa klase eta mezuan zatitu
                    String[] parts = line.split("\t", 2);
                    String clase = parts[0];
                    String mensaje = parts[1];
                    mensaje = mensaje.replace("\"", "\\\"");
                    // Lerroa ARFF fitxategian idatzi
                    writer.write("\"" + mensaje + "\", " + clase + "\n");
                }

                reader.close();
                writer.close();
                ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[2]);
                Instances train = source.getDataSet();
                train.setClassIndex(train.numAttributes()-1);
                System.out.println("train.arff-ak dituen instantzia kopurua: ");
                System.out.println(train.numInstances());

                BufferedReader reader1 = new BufferedReader(new FileReader(args[5]));
                BufferedWriter writer1 = new BufferedWriter(new FileWriter(args[3]));

                // ARFF-aren goiburuak idatzi
                writer1.write("@relation spam\n\n");
                writer1.write("@attribute message string\n\n");
                writer1.write("@attribute klasea {spam, ham}\n");
                writer1.write("@data\n");

                String line1;
                while ((line1 = reader1.readLine()) != null) {
                    // Lerroa klase eta mezuan zatitu
                    String[] parts = line1.split("\t", 2);
                    String clase = parts[0];
                    String mensaje = parts[1];
                    mensaje = mensaje.replace("\"", "\\\"");
                    // Lerroa ARFF fitxategian idatzi
                    writer1.write("\"" + mensaje + "\", " + clase + "\n");
                }

                reader1.close();
                writer1.close();
                ConverterUtils.DataSource source1 = new ConverterUtils.DataSource(args[3]);
                Instances dev = source1.getDataSet();
                dev.setClassIndex(dev.numAttributes()-1);
                System.out.println("dev.arff-ak dituen instantzia kopurua: ");
                System.out.println(dev.numInstances());

                System.out.println("Konbertsioa eginda.");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void karaktereArraroakKendu(String fileName, String fileResult) throws IOException{
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
