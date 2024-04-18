package Aurreprozesamendua;

import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.instance.SparseToNonSparse;

import java.io.*;

public class fssInfoGain {
    public static void main(String[] args) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[0]);
        Instances trainBOW = source.getDataSet();
        //TODO
        trainBOW.setClassIndex(trainBOW.numAttributes()-1);
        //data.setClassIndex(0); --> sin Reorder
        System.out.println("Atributu kopurua trainBOW: " + (trainBOW.numAttributes() - 1));
        AttributeSelection attributeSelection = new AttributeSelection();
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        attributeSelection.setEvaluator(eval);
        Ranker ranker = new Ranker();
        //TODO
        ranker.setNumToSelect(1500);
        attributeSelection.setSearch(ranker);
        attributeSelection.setInputFormat(trainBOW);
        Instances trainBOWFSS = Filter.useFilter(trainBOW, attributeSelection);
        System.out.println("Atributu kopurua berria trainBOWFSS: " + (trainBOWFSS.numAttributes() - 1));
        ArffSaver saver = new ArffSaver();
        saver.setInstances(trainBOWFSS);
        saver.setFile(new File(args[1]));
        saver.writeBatch();

        FileWriter fWriter = new FileWriter(args[3]);
        try
        {
            File file=new File(args[2]);    //creates a new file instance
            FileReader fr=new FileReader(file);   //reads the file
            BufferedReader br=new BufferedReader(fr);  //creates a buffering character input stream
            StringBuffer sb=new StringBuffer();    //constructs a string buffer with no characters
            String line;
            br.readLine();
            for (int i = 0; i < trainBOWFSS.numAttributes()-1; i++) {
                String att = trainBOWFSS.attribute(i).name();
                while((line=br.readLine())!=null) {
                    String lineS = line.split(",")[0];
                    if(lineS.equals(trainBOWFSS.attribute(i).name())) {
                        fWriter.write(line + "\n");
                    }
                }
                fr = new FileReader(file);
                br = new BufferedReader(fr);
            }
        }
        catch(IOException e)
        {
            e.printStackTrace();
        }
        fWriter.close();

        ConverterUtils.DataSource source1 = new ConverterUtils.DataSource(args[4]);
        Instances dev = source1.getDataSet();
        dev.setClassIndex(dev.numAttributes()-1);

        System.out.println("Dev_raw-ren atributu kop: " + dev.numAttributes());

        FixedDictionaryStringToWordVector fixedBoW = new FixedDictionaryStringToWordVector();
        File hiztegiBerria = new File(args[3]);
        fixedBoW.setDictionaryFile(hiztegiBerria);
        fixedBoW.setLowerCaseTokens(false);
        fixedBoW.setTFTransform(false);
        fixedBoW.setIDFTransform(false);
        fixedBoW.setInputFormat(dev);

        Instances dev_bow_fss = Filter.useFilter(dev, fixedBoW);
        System.out.println("Dev_bow_fss-ren atributu kop: " + (dev_bow_fss.numAttributes()-1));

        //TODO
        SparseToNonSparse filter= new SparseToNonSparse();
        filter.setInputFormat(dev_bow_fss);
        dev_bow_fss = Filter.useFilter(dev_bow_fss, filter);

        Reorder reorder = new Reorder();
        reorder.setAttributeIndices("2-" + dev_bow_fss.numAttributes() + ",1");
        reorder.setInputFormat(dev_bow_fss);
        dev_bow_fss = Filter.useFilter(dev_bow_fss, reorder);

        ArffSaver saver2 = new ArffSaver();
        saver2.setInstances(dev_bow_fss);
        saver2.setFile(new File(args[5]));
        saver2.writeBatch();
    }
}