package com;

import io.milvus.client.*;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public class TestConnect {
    @Test(dataProvider = "DefaultConnectArgs", dataProviderClass = MainClass.class)
    public void test_connect(String host, String port) throws ConnectFailedException {
        System.out.println("Host: "+host+", Port: "+port);
        MilvusClient client = new MilvusGrpcClient();
        ConnectParam connectParam = new ConnectParam.Builder()
                .withHost(host)
                .withPort(port)
                .build();
        Response res = client.connect(connectParam);
        assert(res.ok());
        assert(client.isConnected());
    }

    @Test(dataProvider = "DefaultConnectArgs", dataProviderClass = MainClass.class)
    public void test_connect_repeat(String host, String port) {
        MilvusGrpcClient client = new MilvusGrpcClient();
        ConnectParam connectParam = new ConnectParam.Builder()
                .withHost(host)
                .withPort(port)
                .build();
        Response res = null;
        try {
            res = client.connect(connectParam);
            res = client.connect(connectParam);
        } catch (ConnectFailedException e) {
            e.printStackTrace();
        }
        assert (res.ok());
        assert(client.isConnected());
    }

    @Test(dataProvider="InvalidConnectArgs")
    public void test_connect_invalid_connect_args(String ip, String port) {
        MilvusClient client = new MilvusGrpcClient();
        ConnectParam connectParam = new ConnectParam.Builder()
                .withHost(ip)
                .withPort(port)
                .build();
        Response res = null;
        try {
            res = client.connect(connectParam);
        } catch (ConnectFailedException e) {
            e.printStackTrace();
        }
        Assert.assertEquals(res, null);
        assert(!client.isConnected());
    }

    // TODO: MS-615
    @DataProvider(name="InvalidConnectArgs")
    public Object[][] generate_invalid_connect_args() {
        String port = "19530";
        String ip = "";
        return new Object[][]{
                {"1.1.1.1", port},
                {"255.255.0.0", port},
                {"1.2.2", port},
                {"中文", port},
                {"www.baidu.com", "100000"},
                {"127.0.0.1", "100000"},
                {"www.baidu.com", "80"},
        };
    }

    @Test(dataProvider = "DisConnectInstance", dataProviderClass = MainClass.class)
    public void test_disconnect(MilvusClient client, String tableName){
        assert(!client.isConnected());
    }

    @Test(dataProvider = "DisConnectInstance", dataProviderClass = MainClass.class)
    public void test_disconnect_repeatably(MilvusClient client, String tableName){
        Response res = null;
        try {
            res = client.disconnect();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        assert(!res.ok());
        assert(!client.isConnected());
    }
}
