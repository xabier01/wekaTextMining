package Aurreprozesamendua;

import weka.core.Instances;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.converters.ArffLoader;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class hiztegi {
    public static void main(String[] args) throws Exception {
        // ARFF fitxategia kargatu 
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File(args[0]));
        Instances data = loader.getDataSet();

        // Fitxategiko hitzak dituen atributuaren izena lortu
        Attribute wordsAttribute = data.attribute("message");

        // Hiztegia sortu hitzak eta hauen maiztasunak gordetzeko
        Map<String, Integer> dictionary = new HashMap<>();

        // Datu sortaren instantzia guztiak iteratu
        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = data.instance(i);
            // Hitzen atributuaren balioak lortu
            String[] words = instance.stringValue(wordsAttribute).split("\\s+"); // katea hitzetan zatitu

            // Hiztegia eguneratu hitzekin eta bere maiztasunekin
            for (String word : words) {
                // Hitza hiztegian badago, maiztasuna handitu
                if (dictionary.containsKey(word)) {
                    dictionary.put(word, dictionary.get(word) + 1);
                } else { // Hitza hiztegian ez badago, 1 maiztasunarekin gehitu
                    dictionary.put(word, 1);
                }
            }
        }

        // Hiztegia printeatu
        System.out.println("Hitzen hiztegia maiztasunekin:");
        for (Map.Entry<String, Integer> entry : dictionary.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
    }
}
