package Aurreprozesamendua;

import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;


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

        // Guardar el nuevo conjunto de datos transformado en un archivo ARFF
        ArffSaver saver = new ArffSaver();
        saver.setInstances(newData);
        saver.setFile(new File(args[1]));
        saver.writeBatch();

        System.out.println("El archivo ARFF con el filtro StringToWordVector aplicado ha sido guardado exitosamente.");
    }
}
