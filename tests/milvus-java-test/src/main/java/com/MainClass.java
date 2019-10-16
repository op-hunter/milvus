package com;

import io.milvus.client.*;
import org.apache.commons.cli.*;
import org.apache.commons.lang3.RandomStringUtils;
import org.testng.SkipException;
import org.testng.TestNG;
import org.testng.annotations.DataProvider;
import org.testng.xml.XmlClass;
import org.testng.xml.XmlSuite;
import org.testng.xml.XmlTest;

import java.util.ArrayList;
import java.util.List;

public class MainClass {
    private static String host = "127.0.0.1";
    private static String port = "19530";
    public Integer index_file_size = 50;
    public Integer dimension = 128;

    public static void setHost(String host) {
        MainClass.host = host;
    }

    public static void setPort(String port) {
        MainClass.port = port;
    }

    @DataProvider(name="DefaultConnectArgs")
    public static Object[][] defaultConnectArgs(){
        return new Object[][]{{host, port}};
    }

    @DataProvider(name="ConnectInstance")
    public Object[][] connectInstance(){
        MilvusClient client = new MilvusGrpcClient();
        ConnectParam connectParam = new ConnectParam.Builder()
                .withHost(host)
                .withPort(port)
                .build();
        client.connect(connectParam);
        String tableName = RandomStringUtils.randomAlphabetic(10);
        return new Object[][]{{client, tableName}};
    }

    @DataProvider(name="DisConnectInstance")
    public Object[][] disConnectInstance(){
        // Generate connection instance
        MilvusClient client = new MilvusGrpcClient();
        ConnectParam connectParam = new ConnectParam.Builder()
                .withHost(host)
                .withPort(port)
                .build();
        client.connect(connectParam);
        try {
            client.disconnect();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        String tableName = RandomStringUtils.randomAlphabetic(10);
        return new Object[][]{{client, tableName}};
    }

    @DataProvider(name="Table")
    public Object[][] provideTable(){
        Object[][] tables = new Object[2][2];
        MetricType metricTypes[] = { MetricType.L2, MetricType.IP };
        for (Integer i = 0; i < metricTypes.length; ++i) {
            String tableName = metricTypes[i].toString()+"_"+RandomStringUtils.randomAlphabetic(10);
            // Generate connection instance
            MilvusClient client = new MilvusGrpcClient();
            ConnectParam connectParam = new ConnectParam.Builder()
                    .withHost(host)
                    .withPort(port)
                    .build();
            client.connect(connectParam);
            TableSchema tableSchema = new TableSchema.Builder(tableName, dimension)
                    .withIndexFileSize(index_file_size)
                    .withMetricType(metricTypes[i])
                    .build();
            TableSchemaParam tableSchemaParam = new TableSchemaParam.Builder(tableSchema).build();
            Response res = client.createTable(tableSchemaParam);
            if (!res.ok()) {
                System.out.println(res.getMessage());
                throw new SkipException("Table created failed");
            }
            tables[i] = new Object[]{client, tableName};
        }
        return tables;
    }

    public static void main(String[] args) {
        CommandLineParser parser = new DefaultParser();
        Options options = new Options();
        options.addOption("h", "host", true, "milvus-server hostname/ip");
        options.addOption("p", "port", true, "milvus-server port");
        try {
            CommandLine cmd = parser.parse(options, args);
            String host = cmd.getOptionValue("host");
            if (host != null) {
                setHost(host);
            }
            String port = cmd.getOptionValue("port");
            if (port != null) {
                setPort(port);
            }
            System.out.println("Host: "+host+", Port: "+port);
        }
        catch(ParseException exp) {
            System.err.println("Parsing failed.  Reason: " + exp.getMessage() );
        }

//        TestListenerAdapter tla = new TestListenerAdapter();
//        TestNG testng = new TestNG();
//        testng.setTestClasses(new Class[] { TestPing.class });
//        testng.setTestClasses(new Class[] { TestConnect.class });
//        testng.addListener(tla);
//        testng.run();

        XmlSuite suite = new XmlSuite();
        suite.setName("TmpSuite");

        XmlTest test = new XmlTest(suite);
        test.setName("TmpTest");
        List<XmlClass> classes = new ArrayList<XmlClass>();

        classes.add(new XmlClass("com.TestPing"));
        classes.add(new XmlClass("com.TestAddVectors"));
        classes.add(new XmlClass("com.TestConnect"));
        classes.add(new XmlClass("com.TestDeleteVectors"));
        classes.add(new XmlClass("com.TestIndex"));
        classes.add(new XmlClass("com.TestSearchVectors"));
        classes.add(new XmlClass("com.TestTable"));
        classes.add(new XmlClass("com.TestTableCount"));

        test.setXmlClasses(classes) ;

        List<XmlSuite> suites = new ArrayList<XmlSuite>();
        suites.add(suite);
        TestNG tng = new TestNG();
        tng.setXmlSuites(suites);
        tng.run();

    }

}
