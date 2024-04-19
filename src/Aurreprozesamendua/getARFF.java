package Aurreprozesamendua;

import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.*;

public class getARFF{
    public static void main(String[] args) throws Exception{
        try {
            if (args.length != 3){
                System.out.println("Helburua: .txt fitxategi bat (train edo dev) .arff fitxategi formatura egokitzea/pasatzea.");
                System.out.println("Aurre-baldintzak: ");
                System.out.println("    Sartu beharreko argumentuak hurrengoak dira: ");
                System.out.println("    0. txt fitxategiaren path-a.");
                System.out.println("    1. arff fitxategiaren path-a.");
                System.out.println("    2. txt fitxategi garbiaren path-a (karaktere arraroak kenduta gordetzeko).");
                System.out.println("    java -jar getARFF.java \"/path/train.txt\" \"/path/train.arff\" \"/path/trainGarbia.txt\"");
                System.out.println("Post-baldintzak: ");
                System.out.println("    Sartutako .arff fitxategian .txt-ko datuak izatea formatu egokian.");
                System.out.println("    Sartutako 0 argumentua train bada --> \"testua\", klasea --> egituradun .arff fitxategia lortu.");
                System.out.println("    Sartutako 0 argumentua dev bada --> \"testua\", klasea --> egituradun .arff fitxategia lortu.");
                System.out.println("    test (blind) txt fitxategiaren konbertsioa iragarpenak klasean egingo dugu.");
            } else {
                if (args[0].contains("train")) {
                    karaktereArraroakKendu(args[0], args[2]);
                } else if (args[0].contains("dev")) {
                    karaktereArraroakKendu(args[0], args[2]);
                }
                BufferedReader reader = new BufferedReader(new FileReader(args[2]));
                BufferedWriter writer = new BufferedWriter(new FileWriter(args[1]));

                // ARFF-aren goiburuak idatzi
                writer.write("@relation spam\n\n");
                writer.write("@attribute message string\n\n");
                writer.write("@attribute klasea {spam, ham}\n");
                writer.write("@data\n");

                String line;
                while ((line = reader.readLine()) != null) {
                    if (args[0].contains("train")) {
                        // Lerroa klase eta mezuan zatitu
                        String[] parts = line.split("\t", 2);
                        String clase = parts[0];
                        String mensaje = parts[1];
                        mensaje = mensaje.replace("\"", "\\\"");
                        // Lerroa ARFF fitxategian idatzi
                        writer.write("\"" + mensaje + "\", " + clase + "\n");
                    } else if (args[0].contains("dev")) {
                        // Lerroa klase eta mezuan zatitu
                        String[] parts = line.split("\t", 2);
                        String clase = parts[0];
                        String mensaje = parts[1];
                        mensaje = mensaje.replace("\"", "\\\"");
                        // Lerroa ARFF fitxategian idatzi
                        writer.write("\"" + mensaje + "\", " + clase + "\n");

                    /*
                    } else if (args[0].contains("test")) {
                        String[] parts = line.split("\t");
                        // Escribir la línea en el archivo ARFF
                        writer.write("\"" + parts[0] + "\", ?\n");
                        System.out.println(parts[0]);
                    */
                    }
                }

                reader.close();
                writer.close();
                ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[1]);
                Instances data = source.getDataSet();
                data.setClassIndex(data.numAttributes()-1);
                System.out.println("Sortutako .arff-ak dituen instantzia kopurua: ");
                System.out.println(data.numInstances());
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
