package Aurreprozesamendua;

import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.instance.SparseToNonSparse;
import weka.filters.unsupervised.attribute.Reorder;

import java.io.File;

public class arff2bow {
    public static void main(String[] args) throws Exception {
        // Cargar el archivo ARFF
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File(args[0]));
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes()-1);
        System.out.println("klasea: " + data.classAttribute().value(0));
        System.out.println("klasea: " + data.classAttribute().value(1));

        // Aplicar el filtro StringToWordVector
        StringToWordVector filter = new StringToWordVector();
        File hiztegiaFile = new File(args[2]);
        filter.setDictionaryFileToSaveTo(hiztegiaFile);
        filter.setInputFormat(data);

        /*
        // Configurar las características específicas
        filter.setIDFTransform(false);
        filter.setTFTransform(false);
        filter.setAttributeIndices("1");
        filter.setDoNotOperateOnPerClassBasis(false);
        filter.setInvertSelection(false);
        filter.setLowerCaseTokens(true);
        filter.setMinTermFreq(1);
        filter.setNormalizeDocLength(StringToWordVector.None);
        filter.setOutputWordCounts(true);
        filter.setPeriodicPruning(-1.0);
        filter.setStemmer(new weka.core.stemmers.NullStemmer());
        filter.setStopwordsHandler(new weka.core.stopwords.Null());
        filter.setTokenizer(new weka.core.tokenizers.WordTokenizer());
        filter.setUseStoplist(false);
        filter.setWordsToKeep(2000);
        */

        Instances newData = Filter.useFilter(data, filter);


        // me lo crea en sparse --> le tengo que aplicar el filtro non sparse
        SparseToNonSparse filter2 = new SparseToNonSparse();
        filter2.setInputFormat(newData);
        Instances newDataNonSparse = Filter.useFilter(newData,filter2);

        //TODO
        Reorder filter3 = new Reorder();
        filter3.setAttributeIndices("2-"+ newDataNonSparse.numAttributes()+",1");
        filter3.setInputFormat(newDataNonSparse);
        //System.out.println(newDataNonSparse);
        Instances newDataReorder = Filter.useFilter(newDataNonSparse, filter3);
        //newDataReorder.setClassIndex(newDataReorder.numAttributes()-1);
        //System.out.println(newDataReorder);

        System.out.println("Atributu kopurua train: " + (newDataReorder.numAttributes() - 1));

        // Guardar el nuevo conjunto de datos transformado en un archivo ARFF
        ArffSaver saver = new ArffSaver();
        saver.setInstances(newDataReorder);
        saver.setFile(new File(args[1]));
        saver.writeBatch();

        //TODO
        /*
        FixedDictionaryStringToWordVector filterDictionary = new FixedDictionaryStringToWordVector();
        filterDictionary.setDictionaryFile(new File(dictionary));
        filterDictionary.setOutputWordCounts(false);
        filterDictionary.setLowerCaseTokens(true);
        filterDictionary.setInputFormat(devRAW);

        Instances devBOW = Filter.useFilter(devRAW, filterDictionary);
        datuakGorde(devBOWfile, devBOW);
        */

        System.out.println("El archivo ARFF con el filtro StringToWordVector aplicado ha sido guardado exitosamente.");
    }
}
