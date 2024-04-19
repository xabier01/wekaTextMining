package Aurreprozesamendua;

import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.instance.SparseToNonSparse;
import weka.filters.unsupervised.attribute.Reorder;

import java.io.File;

public class arff2bow {
    public static void main(String[] args) throws Exception {
        if (args.length != 3){
            System.out.println("Helburua: train.arff fitxategia Bag of Words errepresentaziora pasatzea (message atributua zenbakizko bektore bihurtzea).");
            System.out.println("          lehenengo hiztegia lortzea");
            System.out.println("Aurre-baldintzak: ");
            System.out.println("    Sartu beharreko argumentuak hurrengoak dira: ");
            System.out.println("    0. train.arff fitxategiaren path-a.");
            System.out.println("    1. trainBOW.arff fitxategiaren path-a.");
            System.out.println("    2. hiztegiaBoW.txt gordetzeko path-a.");
            System.out.println("    java -jar getARFF.java \"/path/train.arff\" \"/path/trainBOW.arff\" \"/path/hiztegiaBoW.txt\"");
            System.out.println("Post-baldintzak: ");
            System.out.println("    trainBOW.arff fitxategia lortzea.");
            System.out.println("    Gure lehenengo hiztegia lortzea (hiztegiaBoW.txt --> hitza, maiztasuna).");
        } else {
            // ARFF fitxategia kargatu
            ArffLoader loader = new ArffLoader();
            loader.setFile(new File(args[0]));
            Instances data = loader.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);
            System.out.println("klasea: " + data.classAttribute().value(0));
            System.out.println("klasea: " + data.classAttribute().value(1));

            // StringToWordVector filtroa aplikatu
            StringToWordVector filter = new StringToWordVector();
            File hiztegiaFile = new File(args[2]);
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
            saver.setFile(new File(args[1]));
            saver.writeBatch();

            System.out.println("ARFF fitxategia StringToWordVector filtroa aplikatuta ondo gorde da.");
        }
    }
}
