package Aurreprozesamendua;

import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.instance.SparseToNonSparse;

import java.io.*;

public class Aurreprozesamendua {
    public static void main(String[] args) throws Exception{
        if (args.length != 11){
            System.out.println("Helburua: Aurreprozesamendua --> trainBOWFSS.arff, devBOWFSS.arff, hiztegiBOWFSS.txt lortzea.");
            System.out.println("Aurre-baldintzak: ");
            System.out.println("    Sartu beharreko argumentuak hurrengoak dira: ");
            System.out.println("    0. train.txt fitxategiaren path-a.");
            System.out.println("    1. dev.txt fitxategiaren path-a.");
            System.out.println("    2. train.arff fitxategiaren path-a.");
            System.out.println("    3. dev.arff fitxategiaren path-a.");
            System.out.println("    4. trainGarbia.txt fitxategi garbiaren path-a (karaktere arraroak kenduta gordetzeko).");
            System.out.println("    5. devGarbia.txt fitxategi garbiaren path-a (karaktere arraroak kenduta gordetzeko).");
            System.out.println("    6. trainBOW.arff fitxategiaren path-a.");
            System.out.println("    7. hiztegiBOWtxt fitxategiaren path-a.");
            System.out.println("    8. trainBOWFSS.arff fitxategiaren path-a.");
            System.out.println("    9. hiztegiaBOWFSS.txt fitxategiaren path-a.");
            System.out.println("    10. devBOWFSS.arff fitxategiaren path-a.");
            System.out.println("    java -jar Aurreprozesamendua.java \"/path/train.txt\" \"/path/dev.txt\" \"/path/train.arff\" \"/path/dev.arff\" \"/path/trainGarbia.txt\" \"/path/devGarbia.txt\" \"/path/trainiBOW.arff\" \"/path/hiztegiBOW.txt\" \"/path/trainBOWFSS.arff\" \"/path/hiztegiBOWFSS.txt\" \"/path/devBOWFSS.arff\"");
            System.out.println("Post-baldintzak: ");
            System.out.println("    trainBOWFSS.arff, devBOWFSS.arff, hiztegiBOWFSS.txt lortzea.");
            System.out.println("    Bidean sortutako fitxategiak gordetzea ikusteko ondo goazen.");
        } else {
            String traintxt = args[0];
            String devtxt = args[1];
            String trainarff = args[2];
            String devarff = args[3];
            String trainGarbiaTxt = args[4];
            String devGarbiaTxt = args[5];
            String trainBOWarff = args[6];
            String hiztegiBOWtxt = args[7];
            String trainBOWFSSarff = args[8];
            String hiztegiaBOWFSStxt = args[9];
            String devBOWFSSarff = args[10];
            getARFF(traintxt,devtxt,trainarff,devarff,trainGarbiaTxt,devGarbiaTxt);
            arff2bow(trainarff, trainBOWarff, hiztegiBOWtxt);
            fssInfoGain(trainBOWarff, trainBOWFSSarff, hiztegiBOWtxt, hiztegiaBOWFSStxt, devarff, devBOWFSSarff);
        }
    }

    private static void getARFF (String traintxt, String devtxt, String trainarff, String devarff, String trainGarbiaTxt, String devGarbiaTxt) throws Exception{

        karaktereArraroakKendu(traintxt, trainGarbiaTxt);
        karaktereArraroakKendu(devtxt, devGarbiaTxt);

        BufferedReader reader = new BufferedReader(new FileReader(trainGarbiaTxt));
        BufferedWriter writer = new BufferedWriter(new FileWriter(trainarff));

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
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(trainarff);
        Instances train = source.getDataSet();
        train.setClassIndex(train.numAttributes()-1);
        System.out.println("train.arff-ak dituen instantzia kopurua: ");
        System.out.println(train.numInstances());

        BufferedReader reader1 = new BufferedReader(new FileReader(devGarbiaTxt));
        BufferedWriter writer1 = new BufferedWriter(new FileWriter(devarff));

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
        ConverterUtils.DataSource source1 = new ConverterUtils.DataSource(devarff);
        Instances dev = source1.getDataSet();
        dev.setClassIndex(dev.numAttributes()-1);
        System.out.println("dev.arff-ak dituen instantzia kopurua: ");
        System.out.println(dev.numInstances());

        System.out.println("Konbertsioa eginda.");
    }

    private static void arff2bow (String trainarff, String trainBOWarff, String hiztegiBOWtxt) throws Exception{
        // ARFF fitxategia kargatu
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File(trainarff));
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        System.out.println("klasea: " + data.classAttribute().value(0));
        System.out.println("klasea: " + data.classAttribute().value(1));

        // StringToWordVector filtroa aplikatu
        StringToWordVector filter = new StringToWordVector();
        File hiztegiaFile = new File(hiztegiBOWtxt);
        filter.setDictionaryFileToSaveTo(hiztegiaFile);
        filter.setInputFormat(data);

        Instances newData = Filter.useFilter(data, filter);

        // me lo crea en sparse --> le tengo que aplicar el filtro non sparse
        SparseToNonSparse filter2 = new SparseToNonSparse();
        filter2.setInputFormat(newData);
        Instances newDataNonSparse = Filter.useFilter(newData, filter2);

        // Reorder --> lehen atributua (klasea) azkenera pasatu
        Reorder filter3 = new Reorder();
        filter3.setAttributeIndices("2-" + newDataNonSparse.numAttributes() + ",1");
        filter3.setInputFormat(newDataNonSparse);
        //System.out.println(newDataNonSparse);
        Instances newDataReorder = Filter.useFilter(newDataNonSparse, filter3);
        //newDataReorder.setClassIndex(newDataReorder.numAttributes()-1);
        //System.out.println(newDataReorder);

        System.out.println("Atributu kopurua train: " + (newDataReorder.numAttributes() - 1));

        // ARFF fitxategi batean gorde transformatutako datu sorta berria
        ArffSaver saver = new ArffSaver();
        saver.setInstances(newDataReorder);
        saver.setFile(new File(trainBOWarff));
        saver.writeBatch();

        System.out.println("ARFF fitxategia StringToWordVector filtroa aplikatuta ondo gorde da.");
    }

    private static void fssInfoGain (String trainBOWarff, String trainBOWFSSarff, String hiztegiaBOWtxt, String hiztegiaBOWFSStxt, String devarff, String devBOWFSSarff) throws Exception{
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(trainBOWarff);
        Instances trainBOW = source.getDataSet();
        trainBOW.setClassIndex(trainBOW.numAttributes() - 1);
        //data.setClassIndex(0); --> sin Reorder
        System.out.println("Atributu kopurua trainBOW: " + (trainBOW.numAttributes() - 1));

        AttributeSelection attributeSelection = new AttributeSelection();
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        attributeSelection.setEvaluator(eval);
        Ranker ranker = new Ranker();
        ranker.setNumToSelect(1500);
        attributeSelection.setSearch(ranker);
        attributeSelection.setInputFormat(trainBOW);
        Instances trainBOWFSS = Filter.useFilter(trainBOW, attributeSelection);
        System.out.println("Atributu kopurua berria trainBOWFSS: " + (trainBOWFSS.numAttributes() - 1));

        ArffSaver saver = new ArffSaver();
        saver.setInstances(trainBOWFSS);
        saver.setFile(new File(trainBOWFSSarff));
        saver.writeBatch();

        FileWriter fWriter = new FileWriter(hiztegiaBOWFSStxt);
        try {
            File file = new File(hiztegiaBOWtxt);
            FileReader fr = new FileReader(file);
            BufferedReader br = new BufferedReader(fr);
            StringBuffer sb = new StringBuffer();
            String line;
            br.readLine();
            for (int i = 0; i < trainBOWFSS.numAttributes() - 1; i++) {
                String att = trainBOWFSS.attribute(i).name();
                while ((line = br.readLine()) != null) {
                    String lineS = line.split(",")[0];
                    if (lineS.equals(trainBOWFSS.attribute(i).name())) {
                        fWriter.write(line + "\n");
                    }
                }
                fr = new FileReader(file);
                br = new BufferedReader(fr);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        fWriter.close();

        ConverterUtils.DataSource source1 = new ConverterUtils.DataSource(devarff);
        Instances dev = source1.getDataSet();
        dev.setClassIndex(dev.numAttributes() - 1);

        System.out.println("Dev_raw-ren atributu kop: " + dev.numAttributes());

        FixedDictionaryStringToWordVector fixedBoW = new FixedDictionaryStringToWordVector();
        File hiztegiBerria = new File(hiztegiaBOWFSStxt);
        fixedBoW.setDictionaryFile(hiztegiBerria);
        fixedBoW.setLowerCaseTokens(false);
        fixedBoW.setTFTransform(false);
        fixedBoW.setIDFTransform(false);
        fixedBoW.setInputFormat(dev);

        Instances dev_bow_fss = Filter.useFilter(dev, fixedBoW);
        System.out.println("Dev_bow_fss-ren atributu kop: " + (dev_bow_fss.numAttributes() - 1));

        SparseToNonSparse filter = new SparseToNonSparse();
        filter.setInputFormat(dev_bow_fss);
        dev_bow_fss = Filter.useFilter(dev_bow_fss, filter);

        Reorder reorder = new Reorder();
        reorder.setAttributeIndices("2-" + dev_bow_fss.numAttributes() + ",1");
        reorder.setInputFormat(dev_bow_fss);
        dev_bow_fss = Filter.useFilter(dev_bow_fss, reorder);

        ArffSaver saver2 = new ArffSaver();
        saver2.setInstances(dev_bow_fss);
        saver2.setFile(new File(devBOWFSSarff));
        saver2.writeBatch();
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
